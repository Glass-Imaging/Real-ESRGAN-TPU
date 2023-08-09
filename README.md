# TPU test training on RealESRGAN
A fork of https://github.com/xinntao/Real-ESRGAN with minimal changes to validate TPU training.

## Install
- Install requirements (Torch and Torchvision commented out to not overwrite versions.)

### Get the data
Largely follows the instructions in the original Readme.
```
# From the project root
mkdir data && cd data
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip # We stick to this only for demo purposes
unzip DIV2K_train_HR.zip && rm DIV2K_train_HR.zip
```

### Generate mock LQ images
The images are simply downscaled by a 4x factor.
```
# From the project root
python3 scripts/generate_mock_lq.py data/DIV2K_train_HR/ data/DIV2K_train_LQ
```

### Generate meta file (list of image paths)
We use paired data for now as it skips the GPU-based degradations, keeping things a little simpler.
> This is no sophisticated SR training - inputs are simply downscaled!
```
# From the project root
python3 scripts/generate_meta_info_pairdata.py --input data/DIV2K_train_HR data/DIV2K_train_LQ/ --meta_info data/meta_info/DIV2K-paired.txt
```
