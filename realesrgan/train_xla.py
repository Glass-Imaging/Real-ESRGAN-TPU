import os

os.environ["PJRT_DEVICE"] = "TPU"  # We use this environment variable as a main check for XLA training
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

from realesrgan.train import main

if __name__ == "__main__":
    USE_XLA_MP = False
    if USE_XLA_MP:
        from torch_xla.distributed import xla_multiprocessing

        xla_multiprocessing.spawn(main, args=(), nprocs=None)
    else:
        main(0)