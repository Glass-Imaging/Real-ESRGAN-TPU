import cv2
import glob
import sys
import os
import os.path as osp
import tqdm


def generate_mock_lq(dirpath_gt: str, dirpath_lq: str):
    if not osp.isdir(dirpath_gt):
        raise ValueError(f"Directory does not exist: {dirpath_gt}")

    os.makedirs(dirpath_lq, exist_ok=False)

    print(f"Generating mock LQ from {dirpath_gt} -> {dirpath_lq}.")

    for path_gt in sorted(glob.glob(osp.join(dirpath_gt, "*.png"))):
        filename = osp.basename(path_gt)
        path_lq = osp.join(dirpath_lq, filename)
        image = cv2.imread(path_gt)
        H, W, _ = image.shape
        image = cv2.resize(image, (W // 4, H // 4))
        cv2.imwrite(path_lq, image)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_mock_lq.py gt_dirpath lq_dirpath")
        quit(-1)

    generate_mock_lq(sys.argv[-2], sys.argv[-1])