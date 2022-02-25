import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import cv2
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


from models import LitImputer
from datasets.kepler_tess import TessDataset, Subset, split_indices
from transforms import Compose, StandardScaler, Mask, RandomCrop, DownSample
from datasets.loading import CollatePred
from utils.postprocessing import eval_full_inputs


def add_arguments(parser):
    """Add program options to provided parser to run the DTST"""

    # loading/training/eval stages
    parser.add_argument('--train_path')
    parser.add_argument(
        '--val_path', help='path to the directory containing the validation samples.')
    parser.add_argument('--test_path', help='path to the directory containing the testing samples.'
                        ' Will activate evaluation on test set.')
    parser.add_argument('--checkpoint')

    # project, checkpoints
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name to be appended to folder's name"
                        + " and Neptune log")
    parser.add_argument("--neptune_project", "--neptune", type=str, default=None,
                        help='name of a neptune project to log the experiment'
                             + 'to. This requires having previously set up a '
                             + 'neptune API token (https://docs.neptune.ai/'
                             + 'getting-started/installation). ')
    parser.add_argument("--tags", nargs="+", type=str,
                        help='additional neptune tags')

    # data options
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help="fraction of samples used for validation.")
    parser.add_argument('--seed', type=int, default=0)

    # Transforms
    parser.add_argument('--crop', default=500, type=int)
    parser.add_argument('--downsample', default=0, type=int)
    parser.add_argument('--skip', default=25, type=int, help="Defines size of the tails skipped"
                        + " when stitching multiple predictions")

    # general training options
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--epochs', "--max_epochs", type=int, default=10000)
    parser.add_argument('--min_epochs', type=int, default=0)
    parser.add_argument('--monitor', help='metrics used '
                        + 'for checkpointing and early stopping')
    parser.add_argument('--patience', type=float)

    # Mask completely at random
    parser.add_argument("--mask_random", type=float, default=0.3,
                        help='Fraction of the input to randomly mask.')
    # Mask in geometric blocks
    parser.add_argument("--mask_geom", type=float, default=None,
                        help='Fraction of the input to mask with geometrically'
                        + ' distributed missing blocks.')
    parser.add_argument("--block_len", type=int, default=5,
                        help='Average masking block lengths.')
    return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    parser = LitImputer.add_model_specific_args(parser)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_workers = args.num_workers
    GPUS = int(torch.cuda.is_available())
    pin_memory = bool(GPUS)
    torch.cuda.empty_cache()
    if num_workers > 0:
        import cv2
        cv2.setNumThreads(0)
    pl.seed_everything(args.seed)

    transform_list = []
    transform_list_2 = []
    if args.crop:
        transform_list += [RandomCrop(args.crop,
                                      exclude_missing_threshold=0.8)]
        transform_list_2 += [RandomCrop(args.crop,
                                        exclude_missing_threshold=0.8)]
    if args.downsample > 1:
        transform_list += [DownSample(args.downsample)]
        transform_list_2 += [DownSample(args.downsample)]
    if args.mask_geom:
        transform_list += [Mask(args.mask_geom, block_len=args.block_len,
                                value=None, exclude_mask=True, max_ratio=0.90)]
    if args.mask_random:
        transform_list += [Mask(args.mask_random, block_len=None,
                                value=None, exclude_mask=True, max_ratio=0.95)]
    transform_list += [StandardScaler(dim=0)]
    transform_list_2 += [StandardScaler(dim=0)]
    transform_both_train = Compose(transform_list)
    transform_both_2 = Compose(transform_list_2)
    transform = None

    train_dataset = TessDataset(args.train_path,
                                load_processed=True,
                                max_samples=args.max_samples,
                                transform=transform,
                                transform_both=transform_both_train,
                                use_cache=True,
                                shuffle=True
                                )

    if args.val_path:
        val_dataset1 = TessDataset(args.val_path,
                                   load_processed=True,
                                   transform=transform,
                                   transform_both=transform_both_train,
                                   use_cache=True,
                                   )
        val_dataset2 = Subset(
            val_dataset1, replace_transform_both=transform_both_2)

        val_loader1 = DataLoader(val_dataset1, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)
        val_loader2 = DataLoader(val_dataset2, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)
    elif args.val_ratio:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    print('train size:', len(train_dataset))
    print('val size:', len(val_dataset1))

    # MODEL DEF AND TRAIN`
    dict_args = vars(args).copy()
    for k in vars(args).keys():
        if dict_args[k] is None:
            dict_args.pop(k)
    lit_model = LitImputer(**dict_args)
    if args.checkpoint:
        lit_model = lit_model.load_from_checkpoint(args.checkpoint)
        print('model loaded from', args.checkpoint)

    if args.neptune_project:
        from pytorch_lightning.loggers import NeptuneLogger
        logger = NeptuneLogger(project=args.neptune_project,
                               name='tess_denoising' if args.name is None else args.name,
                               log_model_checkpoints=False,
                               tags=[str(len(train_dataset))+' samples']
                               + (args.tags if args.tags is not None else [])
                               + (["from ckpt"] if args.checkpoint else []))
    else:
        logger = None

    callbacks = []
    if args.monitor is not None:
        callbacks.append(ModelCheckpoint(monitor=args.monitor,
                                         # dirpath=ckpt_dir,
                                         mode='min',
                                         filename="{epoch:03d}-{"+args.monitor+":.4f}"))

    if args.patience is not None and args.val_ratio:
        callbacks.append(EarlyStopping(args.monitor,
                                       patience=args.patience,
                                       mode='min'))

    trainer = pl.Trainer(max_epochs=args.epochs,
                         min_epochs=args.min_epochs,
                         logger=logger,
                         gpus=GPUS,
                         profiler='simple',
                         callbacks=callbacks,
                         check_val_every_n_epoch=1
                         )
    try:
        result = trainer.fit(lit_model,
                             train_dataloaders=train_loader,
                             val_dataloaders=[val_loader1, val_loader2]
                             )
    except FileNotFoundError as e:
        print(e)

    if args.test_path:
        test_dataset = TessDataset(args.test_path,
                                   load_processed=True,
                                   use_cache=True,
                                   )
        test_dataset1 = Subset(
            test_dataset, replace_transform_both=transform_both_train)
        test_dataset2 = Subset(
            test_dataset, replace_transform_both=transform_both_2)
        test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory)
        test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory)
        loader_pred = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=CollatePred(
                                     args.crop, step=args.crop - args.skip * 2),
                                 num_workers=num_workers, pin_memory=pin_memory)
        torch.cuda.empty_cache()
        trainer.test(lit_model, dataloaders=[test_loader1, test_loader2])

        iqr, dw = eval_full_inputs(
            lit_model, loader_pred, test_dataset, args.skip, 'cuda')
        logger.experiment['testing/full_test_iqr'] = iqr.mean()
        logger.experiment['testing/full_test_dw2'] = np.abs(dw-2).mean()
        print('test IQR', logger.experiment['testing/full_test_iqr'])
        print('test mean absolute DW-2 statistics',
              logger.experiment['testing/full_test_dw2'])
