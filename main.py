import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

from models import LitImputer
from datasets import TessDataset
from transforms import Compose, StandardScaler, Mask, RandomCrop, DownSample


val_ratio = 0.1
batch_size = 64
seed = 0
num_workers = 2
pin_memory = True


if __name__ == '__main__':
    GPUS = int(torch.cuda.is_available())
    torch.cuda.empty_cache()

    # DATA LOADING AND PREP
    transform_both = Compose([RandomCrop(800),
                              DownSample(2),
                              StandardScaler(dim=0),
                              ])

    transform = Compose([Mask(0.3, block_len=None, value=None)])

    if GPUS:
        path = "/state/partition1/mmorvan/data/TESS/lightcurves/0001"
    else:
        path = "/Users/mario/data/TESS/lightcurves/0027"

    train_path = os.path.join(path, 'processed_train')
    test_path = os.path.join(path, 'processed_test')
    if not os.path.exists(train_path):
        train_path = path

    dataset = TessDataset(train_path,
                          load_processed=True,
                          max_samples=8000,
                          transform=transform,
                          transform_both=transform_both,
                          )
    test_dataset = TessDataset(test_path,
                               load_processed=True,
                               transform_both=transform_both,
                               )

    #dataset.n_dim = 1
    # TRAIN/VAL SPLIT
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset,
                                              (train_size, val_size),
                                              generator=torch.Generator().manual_seed(seed))

    print('train size:', len(train_dataset), '\ntest size:',
          len(test_dataset), '\nval size:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    # MODEL DEF AND TRAIN
    torch.manual_seed(seed)
    lit_model = LitImputer(1, d_model=64, dim_feedforward=128,
                           normal_ratio=0.2, keep_ratio=0., token_ratio=0.8)

    logger = NeptuneLogger(project="denoising-transformer",
                           name=str(dataset),
                           log_model_checkpoints=False,
                           tags=(([str(dataset.max_samples)+' samples'] if dataset.max_samples else [])
                                 + [#'bigger-net',
                                    #'attention'
                                    ]))

    trainer = pl.Trainer(max_epochs=1000,
                         logger=logger,
                         gpus=GPUS)
    # , profiler='simple')

    result = trainer.fit(lit_model,
                         train_dataloaders=train_loader,
                         val_dataloaders=val_loader,
                         )

    trainer.test(lit_model, dataloaders=test_loader)
