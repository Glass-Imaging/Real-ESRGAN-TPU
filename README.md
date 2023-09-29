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

## Running
The demo run should work on both GPU (always worked), single-process TPU (just for easier testing) and multi-process TPU (what we actually want.)

### Running on GPU
```
# From the root. CUDA_VISIBLE_DEVICES is needed to disable DDP wrapping for now.
CUDA_VISIBLE_DEVICES=0 python3 -m realesrgan.train -opt options/GPU_paired-data.yml
```

### Running on GPU
```
# From the root. Single/multi process XLA can be set in train_xla.py
python3 -m realesrgan.train_xla -opt options/TPU_paired-data.yml
```