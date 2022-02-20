import os
import glob
import warnings
from typing import (
    Optional,
    Sequence,
)

import numpy as np
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm
from torch._utils import _accumulate
from torch import default_generator, randperm
from torch.utils.data import Dataset, DataLoader


KEPLER_LC_PATTERN = '**/kplr*-*lc.fits'
TESS_LC_PATTERN = '**/tess*-*-*-*-*_*lc.fits'


class DatasetFolder(Dataset):
    """A generic class to load files matching a pattern.

    Samples and targets point to the same object, be wary.

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
        sample = self.get_pretransformed_sample(idx)
        info = self.get_info(idx)

        if self.mask_missing:
            mask = np.isnan(sample)
        else:
            mask = None
        if self.transform_both:
            sample, mask, info = self.transform_both(
                sample, mask=mask, info=info)
        target = sample.copy()
        if self.transform is not None:
            sample, mask, info = self.transform(sample, mask=mask, info=info)
        if self.transform_target is not None:
            target, mask, info = self.transform_target(
                target, mask=mask, info=info)
        return (sample, target, mask, info)

    def get_pretransformed_sample(self, idx):
        if not self.use_cache:
            sample = self.load_lc_file(idx)
            self.cached_data.append(sample)
        else:
            sample = self.cached_data[idx]
        return sample

    def load_lc_file(self, idx):
        # no transform
        processed_file = self.files[idx].replace('.fits', '.pt')
        if self.load_processed and os.path.exists(processed_file):
            out = np.load(processed_file)
        else:
            out = self.load_fits_file(idx)

        return out

    def load_fits_file(self, idx):
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
            warnings.warn(
                'Fewer processed files than original fits files were found in the root directory.')

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
                item = self.load_lc_file(idx)
                with open(processed_file, 'wb') as f:
                    np.save(f, item)

    def save_item(self, idx, save_folder):
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        processed_file = os.path.join(save_folder,
                                      os.path.basename(self.files[idx]).replace('.fits', '.npy'))
        if not os.path.exists(processed_file) or self.overwrite:
            item = self.load_lc_file(idx)
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


class Subset(Dataset):
    def __init__(self, dataset, indices, replace_transform=None, replace_transform_target=None, replace_transform_both=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = replace_transform if replace_transform is not None else self.dataset.transform
        self.transform_target = replace_transform_target if replace_transform_target is not None else self.dataset.transform_target
        self.transform_both = replace_transform_both if replace_transform_both is not None else self.dataset.transform_both

    def __getitem__(self, idx):
        idx = self.indices[idx]
        sample = self.dataset.get_pretransformed_sample(idx)
        info = self.dataset.get_info(idx)

        if self.dataset.mask_missing:
            mask = np.isnan(sample)
        else:
            mask = None
        if self.transform_both:
            sample, mask, info = self.transform_both(
                sample, mask=mask, info=info)
        target = sample.copy()
        if self.transform is not None:
            sample, mask, info = self.transform(sample, mask=mask, info=info)
        if self.transform_target is not None:
            target, mask, info = self.transform_target(
                target, mask=mask, info=info)
        return (sample, target, mask, info)

    def __len__(self):
        return len(self.indices)




def split_indices(lengths: Sequence[int],
                  generator = default_generator):
    r"""
    Randomly split indices into non-overlapping indice subsets.
    Optionally fix the generator for reproducible results, e.g.:

    >>> split_indices([3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    return [indices[offset - length : offset] for offset, length in zip(_accumulate(lengths), lengths)]