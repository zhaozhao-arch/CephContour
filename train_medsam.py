import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.use('TkAgg')

join = os.path.join
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import glob
from PIL import Image
import scipy.ndimage as ndimage

from data_loader import get_loader
from solver import Solver

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

torch.manual_seed(2023)
torch.cuda.empty_cache()


class ImgDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gaus_heatmaps')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, '*.png'), recursive=False))
        self.gt_path_files = [
            file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img = Image.open(join(self.img_path, img_name)).resize((1024, 1024))
        img_npy = np.array(img)[:, :, :3].transpose(2, 0, 1)
        img_npy = (img_npy - img_npy.min()) / np.clip(img_npy.max() - img_npy.min(), a_min=1e-8, a_max=None)

        gt = Image.open(join(self.gt_path, img_name))
        gt = gt.resize((512, 512), resample=Image.Resampling.NEAREST)
        gt_npy = np.array(gt)
        gt_npy = (gt_npy - gt_npy.min()) / np.clip(gt_npy.max() - gt_npy.min(), a_min=1e-8, a_max=None)

        bg_mask = (gt_npy < 0.5).astype(np.float32)
        udf_npy = ndimage.distance_transform_edt(bg_mask)

        return (
            torch.tensor(img_npy).float(),
            torch.tensor(np.array([udf_npy])).float(),
            img_name
        )


def main(args):
    train_dataloader = get_loader(image_path=args.tr_path,
                                  image_size=1024,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  mode='train',
                                  augmentation_prob=args.augmentation_prob)

    test_dataloader = get_loader(image_path=args.test_path,
                                 image_size=1024,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 mode='test',
                                 augmentation_prob=0.)

    solver = Solver(args, train_dataloader, test_dataloader)

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--tr_path', type=str, default='data/imgs')
    parser.add_argument('--test_path', type=str, default='data/test')
    parser.add_argument('--task_name', type=str, default='contourDetect')
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, default='work_dir/curvedetect_model_latest.pth')
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_model_path', type=str, default='')
    parser.add_argument('--work_dir', type=str, default='./work_dir')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument("-use_wandb", type=bool, default=False)
    parser.add_argument("-use_amp", action="store_true", default=False)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    parser.add_argument('--continue_train', type=bool, default=False)
    args = parser.parse_args()

    main(args)