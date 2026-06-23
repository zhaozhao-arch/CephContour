import os
import random
import numpy as np
import scipy.ndimage as ndimage
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from glob import glob
from torch import cat
import torch

join = os.path.join


class DataFolder(data.Dataset):
    def __init__(self, root, image_size=1024, mode='train', augmentation_prob=0.4):
        self.root = root
        self.GT_paths = join(root, 'data/contour')
        image_paths = join(root, 'data/imgs')
        self.image_paths = sorted(glob(join(image_paths, '*.png'), recursive=False))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        filename = os.path.basename(image_path)
        GT_path = glob(join(self.GT_paths, os.path.splitext(filename)[0], '*.png'))

        if not GT_path:
            return self.__getitem__((index + 1) % len(self.image_paths))

        image = Image.open(image_path)
        GTs = [Image.open(gt) for gt in GT_path]

        if len(image.split()) == 4:
            r, g, b, a = image.split()
            image = Image.merge('RGB', (r, g, b))

        aspect_ratio = image.size[1] / image.size[0]
        Transform = []
        ResizeRange = random.randint(1536, 1600)
        Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), antialias=True))
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1 / aspect_ratio

            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
            RotationRange = random.randint(-10, 10)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))
            CropRange = random.randint(1044, 1100)
            Transform.append(T.CenterCrop((int(CropRange + aspect_ratio), CropRange)))
            Transform = T.Compose(Transform)

            image = Transform(image)
            for i in range(len(GTs)):
                GTs[i] = Transform(GTs[i])

            ShiftRange_left = random.randint(0, 20)
            ShiftRange_upper = random.randint(0, 20)
            ShiftRange_right = image.size[0] - random.randint(0, 20)
            ShiftRange_lower = image.size[1] - random.randint(0, 20)
            image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            for i in range(len(GTs)):
                GTs[i] = GTs[i].crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                for i in range(len(GTs)):
                    GTs[i] = F.hflip(GTs[i])

            if random.random() < 0.5:
                image = F.vflip(image)
                for i in range(len(GTs)):
                    GTs[i] = F.vflip(GTs[i])

            Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)
            image = Transform(image)
            Transform = []

        Transform.append(T.Resize((1024, 1024), antialias=True))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        for i in range(len(GTs)):
            GTs[i] = Transform(GTs[i])

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)

        for i in range(len(GTs)):
            GTs[i] = T.Resize((512, 512), antialias=True)(GTs[i])

        GT = cat(GTs)

        gt_np = GT.numpy()
        udf_np = np.zeros_like(gt_np)
        for c in range(gt_np.shape[0]):
            bg_mask = (gt_np[c] < 0.5).astype(np.float32)
            udf_np[c] = ndimage.distance_transform_edt(bg_mask)

        GT = torch.from_numpy(udf_np).float()

        return image, GT

    def __len__(self):
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
    dataset = DataFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader