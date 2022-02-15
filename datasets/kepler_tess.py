import os
import glob
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm

KEPLER_LC_PATTERN = '**/kplr*-*lc.fits'
TESS_LC_PATTERN = '**/tess*-*-*-*-*_*lc.fits'


class DatasetFolder(Dataset):
    """A generic class to load files matching a pattern.

    The class inherits Pytorch utils.data.Dataset class.

    Args:
        root (string): Root directory path.
        processed (boolean, optional): Whether to load the processed
            files when existing in the processed directory. Set to True
            by default.
        save (boolean, optional): Whether to save the processed samples
            as pickled tensors. Set to True by default.
        overwrite (boolean, optional): Whether to overwrite pre-saved
            processed files. Set to True when processed is True or
            False otherwise by default.
        save_folder (string, optional): Path to the directory where
            processed files shall be saved. Set to [root]/processed/ by
            default.
        transform (callable, optional): A function/transform that takes
            in a Tensor and returns a transformed version. E.g,
            ``transforms.RandomCrop``
    Attributes:
        files (list): list of matched files sorted alphabetically.
    """
    FILE_PATTERN = None

    def __init__(self, root, transform=None, transform_target=None, transform_both=None, mask_missing=True, max_samples=None,
                 load_processed=True, check_processed=True, save_processed=None, overwrite=False, save_folder=None, use_cache=False):
        super().__init__()
        self.root = root
        self.transform = transform
        self.transform_target = transform_target
        self.transform_both = transform_both
        self.mask_missing = mask_missing
        self.max_samples = max_samples
        self.load_processed = load_processed
        self.check_processed = check_processed
        self.save_processed = save_processed if save_processed is not None else not load_processed
        self.overwrite = overwrite
        self.save_folder = save_folder
        if save_folder is None:
            self.save_folder = os.path.join(root, 'processed')
        self.use_cache = use_cache
        self.cached_data = []

        # Files discovery, saving and caching
        self.files = None
        self.find_files()
        if self.save_processed:
            self.save_items()

        if self.use_cache:
            self._cache_data()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if not self.use_cache:
            sample = self.load_pretransformed_item(idx)
            self.cached_data.append(sample)
        else:
            sample = self.cached_data[idx]
        info = self.get_info(idx)

        if self.mask_missing:
            mask = np.isnan(sample)
        else:
            mask = None
        if self.transform_both:
            sample, mask = self.transform_both(sample, mask=mask)
            if hasattr(self.transform_both, 'left_crop'):
                info['left_crop'] = self.transform_both.left_crop
        target = sample.copy()
        if self.transform is not None:
            sample, mask = self.transform(sample, mask=mask)
        if self.transform_target is not None:
            target, mask = self.transform_target(target, mask=mask)
        return (sample, target, mask, info)

    def load_pretransformed_item(self, idx):
        # no transform
        processed_file = self.files[idx].replace('.fits', '.pt')
        if self.load_processed and os.path.exists(processed_file):
            out = np.load(processed_file)
        else:
            out = self.load_fits_item(idx)

        return out

    def load_fits_item(self, idx):
        """Fits loader function.
        Note: could be extracted and provided as argument."""
        fits_file = fits.open(self.files[idx])
        fits_data = fits_file[1].data
        out_tensor = np.array(fits_data['PDCSAP_FLUX'].astype(np.float32))[
            :, np.newaxis]
        fits_file.close()
        # out_tensor = np.concatenate(data_columns, axis=1)
        return out_tensor

    def _cache_data(self):
        assert self.use_cache
        loader = DataLoader(
            self,
            num_workers=0,
            shuffle=False,
            batch_size=128
        )
        loader.dataset.use_cache = False
        for _ in tqdm(loader):
            pass
        self.use_cache = True
        print('data successfully cached')

    def find_files(self):
        """Find all files matching a pattern recursively from the root
        directory."""
        file_paths = glob.glob(os.path.join(
            self.root, self.file_pattern), recursive=True)
        self.files = sorted(file_paths)

        if self.load_processed and self.check_processed:
            self._check_processed_completeness()

        if self.max_samples is not None:
            self.files = self.files[:self.max_samples]

    def _check_processed_completeness(self):
        file_paths = glob.glob(os.path.join(
            self.root, self.FILE_PATTERN), recursive=True)
        if len(self.files) < len(file_paths):
            warnings.warn('Fewer processed files than original fits files were found in the root directory.')

    def save_items(self):
        """Save dataset's items as pickled files."""
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if len(self) >= 1000:
            warnings.warn(
                'Saving all items could be slow, to prevent this deactivate saving by setting save=False.')
        for idx in tqdm(range(len(self))):
            processed_file = os.path.join(self.save_folder,
                                          os.path.basename(self.files[idx]).replace('.fits', '.npy'))
            if not os.path.exists(processed_file) or self.overwrite:
                item = self.load_pretransformed_item(idx)
                with open(processed_file, 'wb') as f:
                    np.save(f, item)

    def save_item(self, idx, save_folder):
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        processed_file = os.path.join(save_folder,
                                      os.path.basename(self.files[idx]).replace('.fits', '.npy'))
        if not os.path.exists(processed_file) or self.overwrite:
            item = self.load_pretransformed_item(idx)
            with open(processed_file, 'wb') as f:
                np.save(f, item)

    @property
    def file_pattern(self):
        if self.load_processed:
            return self.FILE_PATTERN.replace('.fits', '.npy')
        else:
            return self.FILE_PATTERN

    def get_info(idx):
        raise NotImplementedError


class KeplerDataset(DatasetFolder):
    FILE_PATTERN = KEPLER_LC_PATTERN


class TessDataset(DatasetFolder):
    FILE_PATTERN = TESS_LC_PATTERN

    def get_info(self, idx):
        fn = Path(self.files[idx]).name
        targetid = int(fn[24:40])
        return {'idx': idx, 'targetid': targetid}
