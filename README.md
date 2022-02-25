# denoising-ts-transformer

Code for submission to [ICRL 2022 AI for Earth and Space Science workshop](https://ai4earthscience.github.io/iclr-2022-workshop/).

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