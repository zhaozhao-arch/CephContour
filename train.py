import os
import sys

sys.path.insert(0, os.getcwd())
import copy
import argparse
import shutil
import time
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils.history import History
from utils.dataloader_contour import Mydataset, collate
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from utils.inference_contour import init_model
from models.build_contour import BuildNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train a two-input model (6 Heads)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--split-validation', default=True, action='store_true', help='whether to split validation set from training set.')
    parser.add_argument('--ratio', type=float, default=0.125, help='the proportion of the validation set to the training set.')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    model_cfg, train_pipeline, val_pipeline_cfg, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    print_info(model_cfg)

    meta = dict()
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('logs', model_cfg.get('backbone').get('type'), dirname)
    meta['save_dir'] = save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seed = init_random_seed(args.seed)
    set_random_seed(seed, deterministic=args.deterministic)
    meta['seed'] = seed

    total_annotations = r"data/6labels.txt"

    with open(total_annotations, encoding='utf-8') as f:
        total_datas = [x for x in f.readlines() if x.strip()]

    if args.split_validation:
        total_nums = len(total_datas)
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)
            rng.shuffle(total_datas)
        val_nums = int(total_nums * args.ratio)
        folds = list(range(int(1.0 / args.ratio)))
        fold = random.choice(folds)
        val_start = val_nums * fold
        val_end = val_nums * (fold + 1)
        train_datas = total_datas[:val_start] + total_datas[val_end:]
        val_datas = total_datas[val_start:val_end]

        val_save_path = os.path.join(save_dir, 'val_split.txt')
        with open(val_save_path, 'w', encoding='utf-8') as f:
            f.writelines(val_datas)
    else:
        train_datas = total_datas.copy()
        test_annotations = 'datas/test.txt'
        with open(test_annotations, encoding='utf-8') as f:
            val_datas = f.readlines()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BuildNet(model_cfg)

    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()

    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'):
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))

    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])
    model.to(device)

    grad_clip_cfg = optimizer_cfg.pop('grad_clip', None)
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)

    lr_update_func = eval(lr_config.pop('type'))(**lr_config)

    train_dataset = Mydataset(train_datas, train_pipeline)
    val_dataset = Mydataset(val_datas, val_pipeline_cfg)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=data_cfg.get('batch_size'),
        num_workers=data_cfg.get('num_workers'),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=data_cfg.get('batch_size'),
        num_workers=data_cfg.get('num_workers'),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate
    )

    runner = dict(
        optimizer=optimizer,
        grad_clip=grad_clip_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        iter=0,
        epoch=0,
        max_epochs=data_cfg.get('train').get('epoches'),
        max_iters=data_cfg.get('train').get('epoches') * len(train_loader),
        best_train_loss=float('INF'),
        best_val_acc=float(0),
        best_train_weight='',
        best_val_weight='',
        last_weight=''
    )
    meta['train_info'] = dict(
        train_loss=[],
        val_loss=[],
        train_acc=[],
        val_acc=[]
    )

    if args.resume_from:
        model, runner, meta = resume_model(model, runner, args.resume_from, meta)
    else:
        shutil.copyfile(args.config, os.path.join(save_dir, os.path.split(args.config)[1]))
        model = init_model(model, data_cfg, device=device, mode='train')

    train_history = History(meta['save_dir'])

    lr_update_func.before_run(runner)

    for epoch in range(runner.get('epoch'), runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)

        train(model, runner, lr_update_func, device, epoch,
              data_cfg.get('train').get('epoches'), data_cfg.get('test'), meta)

        validation(model, runner, data_cfg.get('test'), device, epoch,
                   data_cfg.get('train').get('epoches'), meta)

        t_info = meta['train_info']
        keys_to_check = ['train_loss', 'train_acc', 'val_loss', 'val_acc']

        lengths = [len(t_info[k]) for k in keys_to_check]
        max_len = max(lengths)

        for k in keys_to_check:
            while len(t_info[k]) < max_len:
                if len(t_info[k]) > 0:
                    t_info[k].append(t_info[k][-1])
                else:
                    if 'loss' in k:
                        t_info[k].append(0.0)
                    else:
                        t_info[k].append({'accuracy_top-1': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0})

        try:
            train_history.after_epoch(meta)
        except Exception:
            pass

if __name__ == "__main__":
    main()