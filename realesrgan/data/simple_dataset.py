import os.path as osp
import numpy as np
import cv2
import torch
import glob
from torch.utils import data as data
from torchvision.transforms import ToTensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SimpleDataset(data.Dataset):

    def __init__(self, opt):
        super(SimpleDataset, self).__init__()

        lq_root = opt["lq_path"]
        self.lq_paths = sorted(glob.glob(osp.join(lq_root, "*.png")))
        gt_root = opt["gt_path"]
        self.gt_paths = sorted(glob.glob(osp.join(gt_root, "*.png")))
        assert len(self.lq_paths) == len(self.gt_paths)

        self.gt_size = opt["gt_size"]
        self.lq_size = self.gt_size // opt["scale"]

        self.transform = ToTensor()

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, index):
        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]

        img_lq = cv2.cvtColor(cv2.imread(lq_path), cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)

        img_lq = img_lq[0:self.lq_size, 0:self.lq_size]
        img_gt = img_gt[0:self.gt_size, 0:self.gt_size]

        img_lq = self.transform(img_lq)
        img_gt = self.transform(img_gt)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}