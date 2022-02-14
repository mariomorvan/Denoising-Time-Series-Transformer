import os
import glob
import warnings
import numpy as np
import torch

from astropy.io import fits
from tqdm import tqdm

KEPLER_LC_PATTERN = '**/kplr*-*lc.fits'
TESS_LC_PATTERN = '**/tess*-*-*-*-*_*lc.fits'


class DatasetFolder(torch.utils.data.Dataset):
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

    def __init__(self, root, processed=True, save=True, overwrite=None, save_folder=None, transform=None, num_samples=None):
        super().__init__()
        self.root = root
        self.processed = processed
        self.save = save
        self.overwrite = overwrite
        if overwrite is None:
            # ensures consistency between saved processed data and current dataset by default
            self.overwrite = not self.processed
        self.save_folder = save_folder
        if save_folder is None:
            self.save_folder = os.path.join(root, 'processed')
        self.transform = transform
        self.num_samples = num_samples
        self.files = None
        self.find_files()
        if self.save:
            self.save_items(self.overwrite)

    def find_files(self):
        """Find all files matching a pattern recursively from the root
        directory."""
        file_paths = glob.glob(os.path.join(
            self.root, self.FILE_PATTERN), recursive=True)
        self.files = sorted(file_paths)
        if self.num_samples is not None:
            self.files = self.files[:self.num_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        out_tensor = self._get_pretransformed_item(key)
        if self.transform is not None:
            out_tensor = self.transform(out_tensor)
        return out_tensor

    def _get_pretransformed_item(self, key):
        # no transform
        processed_file = self.files[key].replace('.fits', '.pt')
        if self.processed and os.path.exists(processed_file):
            out_tensor = np.load(processed_file)
        else:
            out_tensor = self._get_fits_item(key)
        return out_tensor

    def _get_fits_item(self, key):
        """Fits loader function.
        Note: could be extracted and provided as argument."""
        fits_file = fits.open(self.files[key])
        fits_data = fits_file[1].data
        out_tensor = np.array(fits_data['PDCSAP_FLUX'].astype(np.float32))[:,np.newaxis]
        fits_file.close()
        # out_tensor = np.concatenate(data_columns, axis=1)
        return out_tensor

    def save_items(self, overwrite=None):
        """Save dataset's items as pickled files."""
        if overwrite is None:
            overwrite = self.overwrite
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if len(self) >= 1000:
            warnings.warn(
                'Saving all items could be slow, to prevent this deactivate saving by setting save=False.')
        for key in tqdm(range(len(self))):
            processed_file = os.path.join(self.save_folder,
                                          os.path.basename(self.files[key]).replace('.fits', '.npy'))
            if not os.path.exists(processed_file) or overwrite:
                item = self._get_pretransformed_item(key)
                with open(processed_file, 'wb') as f:
                    np.save(f, item)


class KeplerDataset(DatasetFolder):
    FILE_PATTERN = KEPLER_LC_PATTERN


class TessDataset(DatasetFolder):
    FILE_PATTERN = TESS_LC_PATTERN