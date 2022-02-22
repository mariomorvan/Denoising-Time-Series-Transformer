import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import cv2

from models import LitImputer
from datasets.kepler_tess import TessDataset, Subset, split_indices
from transforms import Compose, StandardScaler, Mask, RandomCrop, DownSample
from datasets.loading import CollatePred
from utils.postprocessing import eval_full_inputs


max_samples = 1000
val_ratio = 0.1
batch_size = 64
seed = 0
num_workers = 0
pin_memory = True

if __name__ == '__main__':
    GPUS = int(torch.cuda.is_available())
    torch.cuda.empty_cache()
    if num_workers > 0:
        import cv2
        cv2.setNumThreads(0)
    pl.seed_everything(0)

    # DATA LOADING AND PREP
    if GPUS:
        path = "/state/partition1/mmorvan/data/TESS/lightcurves/0001"
    else:
        path = "/Users/mario/data/TESS/lightcurves/0027"

    train_path = os.path.join(path, 'processed_train')
    test_path = os.path.join(path, 'processed_test')

    transform_both_train = Compose([RandomCrop(800, exclude_missing_threshold=0.8),
                                    DownSample(2),
                                    Mask(0.3, block_len=None,
                                         value=None, exclude_mask=True),
                                    StandardScaler(dim=0),
                                    # FillNans(0),
                                    ])

    transform_both_2 = Compose([RandomCrop(800, exclude_missing_threshold=0.8),
                                DownSample(2),
                                #                                Mask(0.3, block_len=None, value=None, exclude_mask=True),
                                StandardScaler(dim=0),
                                ])

    transform = None

    if GPUS:
        path = "/state/partition1/mmorvan/data/TESS/lightcurves/0001"
    else:
        path = "/Users/mario/data/TESS/lightcurves/0027"

    dataset = TessDataset(train_path,
                          load_processed=True,
                          max_samples=max_samples,
                          transform=transform,
                          transform_both=transform_both_train,
                          use_cache=True,
                          )
    test_dataset = TessDataset(test_path,
                               load_processed=True,
                               use_cache=True,
                               )

    #dataset.n_dim = 1
    # TRAIN/VAL SPLIT
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_indices, val_indices = split_indices((train_size, val_size),
                                               generator=torch.Generator().manual_seed(seed))
    train_dataset = Subset(dataset, train_indices)

    val_dataset1 = Subset(dataset, val_indices)
    val_dataset2 = Subset(dataset, val_indices,
                          replace_transform_both=transform_both_2)

    test_dataset1 = Subset(
        test_dataset, replace_transform_both=transform_both_train)
    test_dataset2 = Subset(
        test_dataset, replace_transform_both=transform_both_2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader1 = DataLoader(val_dataset1, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader2 = DataLoader(val_dataset2, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    loader_pred = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=CollatePred(400, step=350),
                             num_workers=num_workers, pin_memory=pin_memory)

    print('train size:', len(train_dataset), '\ntest size:',
          len(test_dataset1), '\nval size:', len(val_dataset1))

    # MODEL DEF AND TRAIN
    torch.manual_seed(seed)
    lit_model = LitImputer(n_dim=1, d_model=64, dim_feedforward=128, num_layers=3, lr=0.001,
                           random_ratio=0.1, token_ratio=0.9,
                           train_unit='noise', train_loss='mae')
    from pytorch_lightning.loggers import NeptuneLogger
    logger = NeptuneLogger(project="denoising-transformer",
                           name='tess_denoising',
                           log_model_checkpoints=False,
                           tags=[str(len(dataset))+' samples',
                                 "train - " + lit_model.train_unit,
                                 "0.2 geom 0.1 bernouille"
                                 ])

    trainer = pl.Trainer(max_epochs=5,
                         logger=logger,
                         gpus=GPUS,
                         profiler='simple',
                         check_val_every_n_epoch=1
                         )
    try:
        result = trainer.fit(lit_model,
                             train_dataloaders=train_loader,
                             val_dataloaders=[val_loader1, val_loader2]
                             )
    except FileNotFoundError as e:
        print(e)

    trainer.test(lit_model, dataloaders=[test_loader1, test_loader2])

    iqr, dw = eval_full_inputs(
        lit_model, loader_pred, test_dataset, 25, 'cuda')
    logger.experiment['testing/full_test_iqr'] = iqr
    logger.experiment['testing/full_test_dw2'] = dw
    print(iqr, dw)
