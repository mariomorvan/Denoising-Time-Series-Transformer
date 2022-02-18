import numpy as np
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, size: int, seq_len: int = 300, cadence=0.02, target_mode='self', sigma_noise=1e-4,
                 transform=None,  transform_target=None, transform_both=None, seed=None):
        super().__init__()
        self.size = int(size)
        self.seq_len = seq_len
        self.cadence = cadence
        self.target_mode = target_mode
        self.sigma_noise = sigma_noise
        self.transform = transform
        self.transform_target = transform_target
        self.transform_both = transform_both
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.data = []
        self._fill_data()

    def set_target_mode(self, target_mode):
        assert target_mode in ['self', 'clean']
        self.target_mode = target_mode

    def _fill_data(self):
        # time in days assuming 30min cadence
        self.time = np.linspace(0, self.seq_len * self.cadence, self.seq_len)

        for _ in range(self.size):
            a = np.random.uniform(-1, 1)
            b = np.random.uniform(-0.1, 0.1)
            w = np.random.uniform(1, 5)
            offset = np.random.uniform(-self.seq_len * self.cadence, 0)
            trend = 1 + ((np.sin(w * (self.time+offset)) +
                          a * self.time + b * self.time**2) / 500)
            noise = np.random.normal(0, self.sigma_noise, self.seq_len)
            observed_flux = (trend + noise)[:, np.newaxis].astype(np.float32)
            if self.target_mode == 'self':
                target = observed_flux.copy()
            elif self.target_mode == 'clean':
                target = trend[:, np.newaxis].astype(np.float32)
            else:
                raise NotImplementedError
            self.data += [(observed_flux, target)]

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        sample, target = self.data[key]
        mask = None
        info = {'idx': key}
        if self.transform_both:
            sample, mask, info = self.transform_both(sample, mask=mask, info=info)
        if self.transform is not None:
            sample, mask, info = self.transform(sample, mask=mask, info=info)
        if self.transform_target is not None:
            target, mask, info = self.transform_target(target, mask=mask, info=info)
        return (sample, target, mask, info)
