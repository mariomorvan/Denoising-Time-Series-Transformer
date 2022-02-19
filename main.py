import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import cv2

from models import LitImputer
from datasets import TessDataset
from transforms import Compose, StandardScaler, Mask, RandomCrop, DownSample


max_samples = 1000
val_ratio = 0.1
batch_size = 64
seed = 0
num_workers = 0
pin_memory = True

if num_workers > 0:
    import cv2
    cv2.setNumThreads(0)

if __name__ == '__main__':
    GPUS = int(torch.cuda.is_available())
    torch.cuda.empty_cache()

    # DATA LOADING AND PREP

    transform_both_train = Compose([RandomCrop(800, exclude_missing_threshold=0.8),
                                    DownSample(2),
                                    Mask(0.3, block_len=None,
                                         value=None, exclude_mask=True),
                                    StandardScaler(dim=0),
                                    # FillNans(0),
                                    ])

    transform_both_test = Compose([RandomCrop(800, exclude_missing_threshold=0.8),
                                   DownSample(2),
                                   #                                Mask(0.3, block_len=None, value=None, exclude_mask=True),
                                   StandardScaler(dim=0),
                                   # FillNans(0),
                                   ])

    transform = None

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
                          max_samples=max_samples,
                          transform=transform,
                          transform_both=transform_both_train,
                          use_cache=True,
                          )
    test_dataset = TessDataset(test_path,
                               load_processed=True,
                               transform_both=transform_both_test,
                               use_cache=True,
                               )

    # TRAIN/VAL SPLIT
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset,
                                              (train_size, val_size),
                                              generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print('train size:', len(train_dataset), '\ntest size:',
          len(test_dataset), '\nval size:', len(val_dataset))

    # MODEL DEF AND TRAIN
    torch.manual_seed(seed)
    lit_model = LitImputer(n_dim=1, d_model=64, dim_feedforward=128, lr=0.001,
                           # attention='linear', seq_len=400,
                           random_ratio=1, zero_ratio=0., keep_ratio=0., token_ratio=0,
                           train_unit='standard')
    from pytorch_lightning.loggers import NeptuneLogger
    logger = NeptuneLogger(project="denoising-transformer",
                           name='tess_denoising',
                           log_model_checkpoints=True,
                           tags=[str(len(dataset))+' samples',
                                 "train - " + lit_model.train_unit,
                                 ])

    trainer = pl.Trainer(max_epochs=3000,
                         logger=logger,
                         gpus=GPUS,
                         profiler='simple')

    result = trainer.fit(lit_model,
                         train_dataloaders=train_loader,
                         val_dataloaders=val_loader,
                         )

    trainer.test(lit_model, dataloaders=test_loader)
