# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import cv2
import numpy as np

from .build import PIPELINES
from PIL import Image


@PIPELINES.register_module()
class LoadDoubleImageFromFile:

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def get(self, filepath):
        with open(filepath, 'rb') as f:
            img = Image.open(f)
            value_buf = img.convert('RGB')
        return np.array(value_buf)

    def __call__(self, results):
        if 'filename1' in results['img_info']:
            img1_path = osp.join(results['img_prefix'], results['img_info']['filename1']) if results['img_prefix'] else results['img_info']['filename1']
            img1 = self.get(img1_path)
            if self.to_float32:
                img1 = img1.astype(np.float32)
            results['img'] = img1
            results['img_shape'] = img1.shape[:2]
        elif 'filename2' in results['img_info']:
            img2_path = osp.join(results['img_prefix'], results['img_info']['filename2']) if results['img_prefix'] else results['img_info']['filename2']
            img2 = self.get(img2_path)
            if self.to_float32:
                img2 = img2.astype(np.float32)
            results['img'] = img2
            results['img_shape'] = img2.shape[:2]
        else:
            raise KeyError("img_info must contain 'filename1' or 'filename2'")

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(to_float32={self.to_float32})"
