import numpy as np
import torch
import cv2
from core.visualization import imshow_infos
from core.datasets.compose import Compose
from utils.checkpoint import load_checkpoint

def init_model(model, data_cfg, device='cuda:0', mode='eval'):
    if mode == 'train':
        if data_cfg.get('train').get('pretrained_flag') and data_cfg.get('train').get('pretrained_weights'):
            print('Loading {}'.format(data_cfg.get('train').get('pretrained_weights').split('/')[-1]))
            load_checkpoint(model, data_cfg.get('train').get('pretrained_weights'), device, False)
    elif mode == 'eval':
        print('Loading {}'.format(data_cfg.get('test').get('ckpt').split('/')[-1]))
        model.eval()
        load_checkpoint(model, data_cfg.get('test').get('ckpt'), device, False)
    model.to(device)
    return model

def inference_model(model, image1, image2, val_pipeline):
    if isinstance(image1, str):
        if val_pipeline[0]['type'] != 'LoadDoubleImageFromFile':
            val_pipeline1 = val_pipeline.copy()
            val_pipeline1.insert(0, dict(type='LoadDoubleImageFromFile'))
        else:
            val_pipeline1 = val_pipeline
        data1 = {
            'img_info': dict(filename1=image1),
            'img_prefix': None,
            'filename': image1
        }
    else:
        if val_pipeline[0]['type'] == 'LoadDoubleImageFromFile':
            val_pipeline1 = val_pipeline.copy()
            val_pipeline1.pop(0)
        else:
            val_pipeline1 = val_pipeline
        data1 = {
            'img': image1,
            'filename': None
        }

    if isinstance(image2, str):
        if val_pipeline[0]['type'] != 'LoadDoubleImageFromFile':
            val_pipeline2 = val_pipeline.copy()
            val_pipeline2.insert(0, dict(type='LoadDoubleImageFromFile'))
        else:
            val_pipeline2 = val_pipeline
        data2 = {
            'img_info': dict(filename2=image2),
            'img_prefix': None,
            'filename': image2
        }
    else:
        if val_pipeline[0]['type'] == 'LoadDoubleImageFromFile':
            val_pipeline2 = val_pipeline.copy()
            val_pipeline2.pop(0)
        else:
            val_pipeline2 = val_pipeline
        data2 = {
            'img': image2,
            'filename': None
        }

    pipeline = Compose(val_pipeline1)
    img1 = pipeline(data1)['img'].unsqueeze(0)

    pipeline = Compose(val_pipeline2)
    img2 = pipeline(data2)['img'].unsqueeze(0)

    device = next(model.parameters()).device

    with torch.no_grad():
        scores = model(img1.to(device), img2.to(device), return_loss=False)
        results = []
        num_branches = scores.size(1)

        for branch_idx in range(num_branches):
            branch_scores = scores[:, branch_idx, :]
            pred_score, pred_label = torch.max(branch_scores, axis=1)
            branch_result = {
                'branch': branch_idx,
                'pred_label': pred_label.item(),
                'pred_score': float(pred_score),
                'pred_class': pred_label.item()
            }
            results.append(branch_result)
    return results

def inference_backbone(model, image1, image2, val_pipeline):
    if isinstance(image1, str):
        if val_pipeline[0]['type'] != 'LoadImageFromFile':
            val_pipeline1 = val_pipeline.copy()
            val_pipeline1.insert(0, dict(type='LoadImageFromFile'))
        else:
            val_pipeline1 = val_pipeline
        data1 = dict(img_info=dict(filename=image1), img_prefix=None)
    else:
        if val_pipeline[0]['type'] == 'LoadImageFromFile':
            val_pipeline1 = val_pipeline.copy()
            val_pipeline1.pop(0)
        else:
            val_pipeline1 = val_pipeline
        data1 = dict(img=image1, filename=None)

    if isinstance(image2, str):
        if val_pipeline[0]['type'] != 'LoadImageFromFile':
            val_pipeline2 = val_pipeline.copy()
            val_pipeline2.insert(0, dict(type='LoadImageFromFile'))
        else:
            val_pipeline2 = val_pipeline
        data2 = dict(img_info=dict(filename=image2), img_prefix=None)
    else:
        if val_pipeline[0]['type'] == 'LoadImageFromFile':
            val_pipeline2 = val_pipeline.copy()
            val_pipeline2.pop(0)
        else:
            val_pipeline2 = val_pipeline
        data2 = dict(img=image2, filename=None)

    pipeline = Compose(val_pipeline1)
    img1 = pipeline(data1)['img'].unsqueeze(0)

    pipeline = Compose(val_pipeline2)
    img2 = pipeline(data2)['img'].unsqueeze(0)

    device = next(model.parameters()).device

    with torch.no_grad():
        img_feats = model.backbone(img1.to(device))
        contour_feats = model.contour_backbone(img2.to(device))
        fused_feats = model.neck((img_feats, contour_feats))
        return fused_feats

def show_result(img, result, text_color='white', font_scale=0.5, row_width=20, show=False, fig_size=(15, 10), win_name='', wait_time=0, out_file=None):
    img = cv2.imread(img)
    img = img.copy()
    img = imshow_infos(img, result, text_color=text_color, font_size=int(font_scale * 50), row_width=row_width, win_name=win_name, show=show, fig_size=fig_size, wait_time=wait_time, out_file=out_file)
    return img

def show_result_pyplot(model, img, result, fig_size=(15, 10), title='result', wait_time=0, out_file=None):
    if hasattr(model, 'module'):
        model = model.module
    show_result(img, result, show=True, fig_size=fig_size, win_name=title, wait_time=wait_time, out_file=out_file)