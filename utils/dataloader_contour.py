import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from core.datasets.compose import Compose


class Mydataset(Dataset):
    def __init__(self, gt_labels, cfg):
        self.gt_labels = gt_labels
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)
        self.data_infos = self.load_annotations()

    def __len__(self):
        return len(self.gt_labels)

    def __getitem__(self, index):
        data_info = copy.deepcopy(self.data_infos[index])
        seed = np.random.randint(0, np.iinfo(np.int32).max)

        img1_data = {
            'img_prefix': data_info['img_prefix'],
            'img_info': {'filename1': data_info['img_info']['filename1']},
            'gt_label': data_info['gt_label'],
            'filename': data_info['img_info']['filename1']
        }
        np.random.seed(seed)
        img1_results = self.pipeline(img1_data)

        img2_data = {
            'img_prefix': data_info['img_prefix'],
            'img_info': {'filename2': data_info['img_info']['filename2']},
            'gt_label': data_info['gt_label'],
            'filename': data_info['img_info']['filename2']
        }
        np.random.seed(seed)
        img2_results = self.pipeline(img2_data)

        return (
            img1_results['img'],
            img2_results['img'],
            data_info['gt_label'],
            (data_info['img_info']['filename1'], data_info['img_info']['filename2'])
        )

    def load_annotations(self):
        if len(self.gt_labels) == 0:
            raise TypeError('ann_file is None')

        samples = [x.strip().split() for x in self.gt_labels]
        data_infos = []
        for items in samples:
            filename1 = items[0]
            filename2 = items[1]
            labels = items[2:]

            info = {'img_prefix': None}
            info['img_info'] = {
                'filename1': filename1,
                'filename2': filename2,
                'labels': list(map(int, labels))
            }
            info['gt_label'] = np.array(labels, dtype=np.int64)
            data_infos.append(info)
        return data_infos


def collate(batches):
    img1s, img2s, gts, image_paths = tuple(zip(*batches))
    img1s = torch.stack(img1s, dim=0)
    img2s = torch.stack(img2s, dim=0)
    gts = torch.stack([torch.as_tensor(gt) for gt in gts], 0)
    return img1s, img2s, gts, image_paths