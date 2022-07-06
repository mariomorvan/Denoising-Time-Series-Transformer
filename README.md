# Don’t Pay Attention to the Noise: Learning Self-supervised Light Curve Representations with a Denoising Time Series Transformer

This is the official repo associated with the above work presented at ICLR 2022 [AI for Earth & Space Science](https://ai4earthscience.github.io/iclr-2022-workshop/accepted) and ICML 2022 [Machine Learning for Astrophysics](https://ml4astro.github.io/icml2022/) workshops.

It contains some utilities to process Light Curve Datasets, as weel as the Denoising Time Series Transformer implemented as a Pytorch lightning module.

We advise to run the code in a dedicated virtual/conda environment after installing the required dependencies:

```bash
pip install -r requirements
```

TESS data used for experiments can be downloaded by executing the bash script ```tesscurl_sector_1_lc.sh``` saved in the data directory (and found on [STSCI's archive website](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html))

Finally the DTST can be run with:
```python 
main.py --train_path [path_to_training_directory]
```
See options with ```python main.py --help```