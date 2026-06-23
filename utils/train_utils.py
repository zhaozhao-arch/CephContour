import os
import torch
import torch.distributed as dist
import sys
import types
import importlib
import random
from tqdm import tqdm
import numpy as np
from terminaltables import AsciiTable
from torch.optim import Optimizer
from core.evaluations import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.common import get_dist_info


def init_random_seed(seed=None, device='cuda'):
    if seed is not None:
        return seed
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed
    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def file2dict(filename):
    (path, file) = os.path.split(filename)
    abspath = os.path.abspath(os.path.expanduser(path))
    sys.path.insert(0, abspath)
    mod = importlib.import_module(file.split('.')[0])
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
           and not isinstance(value, types.ModuleType)
           and not isinstance(value, types.FunctionType)
    }
    return cfg_dict.get('model_cfg'), cfg_dict.get('train_pipeline'), cfg_dict.get('val_pipeline'), cfg_dict.get(
        'data_cfg'), cfg_dict.get('lr_config'), cfg_dict.get('optimizer_cfg')


def print_info(cfg):
    backbone = cfg.get('backbone').get('type') if cfg.get('backbone') is not None else 'None'
    if isinstance(cfg.get('neck'), list):
        temp = []
        lists = cfg.get('neck')
        for i in lists:
            temp.append(i.get('type'))
        neck = ' '.join(temp)
    else:
        neck = cfg.get('neck').get('type') if cfg.get('neck') is not None else 'None'

    head_cfg = cfg.get('head')
    if isinstance(head_cfg, list):
        head = f"List of {len(head_cfg)} heads ({head_cfg[0].get('type')})"
        loss = head_cfg[0].get('loss').get('type')
    else:
        head = head_cfg.get('type') if head_cfg is not None else 'None'
        loss = head_cfg.get('loss').get('type') if head_cfg is not None and head_cfg.get('loss') is not None else 'None'

    TITLE = 'Model info'
    TABLE_DATA = (
        ('Backbone', 'Neck', 'Head', 'Loss'),
        (backbone, neck, head, loss)
    )
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()


def get_info(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    names = []
    indexs = []
    for data in class_names:
        name, index = data.split(' ')
        names.append(name)
        indexs.append(int(index))
    return names, indexs


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def resume_model(model, runner, checkpoint, meta, resume_optimizer=True, map_location='default'):
    if map_location == 'default':
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            checkpoint = load_checkpoint(
                model,
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = load_checkpoint(model, checkpoint)
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_location)

    runner['epoch'] = checkpoint['meta']['epoch']
    runner['iter'] = checkpoint['meta']['iter']
    runner['best_train_weight'] = checkpoint['meta'].get('best_train_weight', '')
    runner['last_weight'] = checkpoint['meta'].get('last_weight', '')
    runner['best_val_weight'] = checkpoint['meta'].get('best_val_weight', '')
    runner['best_train_loss'] = checkpoint['meta'].get('best_train_loss', float('inf'))
    runner['best_val_acc'] = checkpoint['meta'].get('best_val_acc', 0.0)

    if meta is None:
        meta = {}
    meta = checkpoint['meta']

    if 'optimizer' in checkpoint and resume_optimizer:
        if isinstance(runner['optimizer'], Optimizer):
            runner['optimizer'].load_state_dict(checkpoint['optimizer'])
        elif isinstance(runner['optimizer'], dict):
            for k in runner['optimizer'].keys():
                runner.optimizer[k].load_state_dict(checkpoint['optimizer'][k])
        else:
            raise TypeError(f"Optimizer should be dict or torch.optim.Optimizer but got {type(runner.optimizer)}")

    print('resumed epoch %d, iter %d' % (runner['epoch'], runner['iter']))
    return model, runner, meta


def train(model, runner, lr_update_func, device, epoch, epoches, test_cfg, meta):
    train_loss = 0
    pred_list, target_list = [], []
    runner['epoch'] = epoch + 1
    meta['epoch'] = runner['epoch']

    model.train()
    with tqdm(total=len(runner.get('train_loader')), desc=f'Train: Epoch {epoch + 1}/{epoches}', postfix=dict,
              mininterval=0.3) as pbar:
        for iter, batch in enumerate(runner.get('train_loader')):
            image1, image2, targets, _ = batch

            with torch.no_grad():
                image1 = image1.to(device)
                image2 = image2.to(device)
                targets = targets.to(device)
                target_list.append(targets)

            if targets.dim() > 1:
                targets = targets.squeeze()

            runner.get('optimizer').zero_grad()
            lr_update_func.before_train_iter(runner)

            preds, losses = model(image1, image2, targets=targets, return_loss=True, train_statu=True)
            total_loss = sum(tensor for name, tensor in losses.items() if 'loss' in name)
            total_loss.backward()

            if runner.get('grad_clip') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), **runner['grad_clip'])

            runner.get('optimizer').step()

            pred_list.append(preds)
            train_loss += total_loss.item()

            pbar.set_postfix(**{'Loss': train_loss / (iter + 1), 'Lr': get_lr(runner.get('optimizer'))})
            runner['iter'] += 1
            meta['iter'] = runner['iter']
            pbar.update(1)

    all_preds = torch.cat(pred_list)
    all_targets = torch.cat(target_list)
    eval_results = {}
    temp_acc, temp_prec, temp_rec, temp_f1 = [], [], [], []

    for i in range(6):
        pred_i = all_preds[:, i, :]
        target_i = all_targets[:, i]
        res = evaluate(pred_i, target_i, test_cfg.get('metrics'), test_cfg.get('metric_options'))

        for k, v in res.items():
            new_key = k.replace('accuracy', f'accuracy_head{i}')
            if 'top-1' in k: new_key = f'accuracy_head{i}_top1'
            eval_results[new_key] = v

        acc = res.get('accuracy_top-1', 0.0)
        prec = res.get('precision', 0.0)
        rec = res.get('recall', 0.0)
        f1 = res.get('f1_score', 0.0)

        def to_float(x):
            if isinstance(x, (list, tuple, np.ndarray, torch.Tensor)):
                return float(x[0]) if len(x) > 0 else 0.0
            return float(x)

        temp_acc.append(to_float(acc))
        temp_prec.append(np.mean(prec) if isinstance(prec, (list, np.ndarray)) and len(prec) > 1 else to_float(prec))
        temp_rec.append(np.mean(rec) if isinstance(rec, (list, np.ndarray)) and len(rec) > 1 else to_float(rec))
        temp_f1.append(np.mean(f1) if isinstance(f1, (list, np.ndarray)) and len(f1) > 1 else to_float(f1))

    eval_results['accuracy_top-1'] = np.mean(temp_acc)
    eval_results['precision'] = np.mean(temp_prec)
    eval_results['recall'] = np.mean(temp_rec)
    eval_results['f1_score'] = np.mean(temp_f1)

    meta['train_info']['train_loss'].append(train_loss / (iter + 1))
    meta['train_info']['train_acc'].append(eval_results)

    if train_loss / len(runner.get('train_loader')) < runner.get('best_train_loss'):
        runner['best_train_loss'] = train_loss / len(runner.get('train_loader'))
        meta['best_train_loss'] = runner['best_train_loss']
        if epoch > 0 and os.path.isfile(runner['best_train_weight']):
            os.remove(runner['best_train_weight'])
        runner['best_train_weight'] = os.path.join(meta['save_dir'], 'Train_Epoch{:03}-Loss{:.3f}.pth'.format(epoch + 1,
                                                                                                              train_loss / len(
                                                                                                                  runner.get(
                                                                                                                      'train_loader'))))
        meta['best_train_weight'] = runner['best_train_weight']
        save_checkpoint(model, runner.get('best_train_weight'), runner.get('optimizer'), meta)

    TITLE = 'Train Results'
    NUM_HEADS = 6
    headers = [f'Branch {i} Top-1 Acc' for i in range(NUM_HEADS)]
    branch_accs = ['{:.2f}'.format(x) for x in temp_acc]
    TABLE_DATA = (tuple(headers), tuple(branch_accs))
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()


def validation(model, runner, cfg, device, epoch, epoches, meta):
    pred_list, target_list = [], []
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(runner.get('val_loader')), desc=f'Test : Epoch {epoch + 1}/{epoches}', postfix=dict,
                  mininterval=0.3) as pbar:
            for iter, batch in enumerate(runner.get('val_loader')):
                image1, image2, targets, _ = batch
                if targets.dim() > 1:
                    targets = targets.squeeze()

                preds, losses = model(image1.to(device), image2.to(device), targets=targets.to(device),
                                      return_loss=True, train_statu=True)
                loss_items = [tensor for name, tensor in losses.items() if 'loss' in name and 'acc' not in name]
                total_loss = sum(loss_items) / len(loss_items)

                pred_list.append(preds)
                target_list.append(targets.to(device))
                val_loss += total_loss.item()
                pbar.set_postfix(**{'Loss': val_loss / (iter + 1)})
                pbar.update(1)

    all_preds = torch.cat(pred_list)
    all_targets = torch.cat(target_list)
    eval_results = {}
    temp_acc, temp_prec, temp_rec, temp_f1 = [], [], [], []

    for i in range(6):
        pred_i = all_preds[:, i, :]
        target_i = all_targets[:, i]
        res = evaluate(pred_i, target_i, cfg.get('metrics'), cfg.get('metric_options'))

        for k, v in res.items():
            if 'top-1' in k:
                new_key = f'accuracy_head{i}_top1'
            else:
                new_key = k.replace('accuracy', f'accuracy_head{i}')
            eval_results[new_key] = v

        acc = res.get('accuracy_top-1', 0.0)
        prec = res.get('precision', 0.0)
        rec = res.get('recall', 0.0)
        f1 = res.get('f1_score', 0.0)

        def to_float(x):
            if isinstance(x, (list, tuple, np.ndarray, torch.Tensor)):
                return float(x[0]) if len(x) > 0 else 0.0
            return float(x)

        temp_acc.append(to_float(acc))
        temp_prec.append(np.mean(prec) if isinstance(prec, (list, np.ndarray)) and len(prec) > 1 else to_float(prec))
        temp_rec.append(np.mean(rec) if isinstance(rec, (list, np.ndarray)) and len(rec) > 1 else to_float(rec))
        temp_f1.append(np.mean(f1) if isinstance(f1, (list, np.ndarray)) and len(f1) > 1 else to_float(f1))

    eval_results['accuracy_top-1'] = np.mean(temp_acc)
    eval_results['precision'] = np.mean(temp_prec)
    eval_results['recall'] = np.mean(temp_rec)
    eval_results['f1_score'] = np.mean(temp_f1)

    meta['train_info']['val_acc'].append(eval_results)
    meta['train_info']['val_loss'].append(val_loss / (iter + 1))

    TITLE = 'Validation Results'
    NUM_HEADS = 6
    headers = [f'Branch {i} Top-1 Acc' for i in range(NUM_HEADS)]
    branch_accs = ['{:.2f}'.format(x) for x in temp_acc]
    TABLE_DATA = (tuple(headers), tuple(branch_accs))
    table_instance = AsciiTable(TABLE_DATA, TITLE)
    print()
    print(table_instance.table)
    print()

    avg_acc = np.mean(temp_acc)

    if avg_acc > runner.get('best_val_acc', 0.0):
        runner['best_val_acc'] = avg_acc
        meta['best_val_acc'] = runner['best_val_acc']
        if epoch > 0 and os.path.isfile(runner.get('best_val_weight', '')):
            os.remove(runner['best_val_weight'])
        runner['best_val_weight'] = os.path.join(meta['save_dir'], f'Val_Epoch{epoch + 1:03}-AvgAcc{avg_acc:.3f}.pth')
        meta['best_val_weight'] = runner['best_val_weight']
        save_checkpoint(model, runner['best_val_weight'], runner.get('optimizer'), meta)

    if epoch > 0 and os.path.isfile(runner.get('last_weight', '')):
        os.remove(runner['last_weight'])

    runner['last_weight'] = os.path.join(meta['save_dir'], f'Last_Epoch{epoch + 1:03}.pth')
    meta['last_weight'] = runner['last_weight']
    save_checkpoint(model, runner['last_weight'], runner.get('optimizer'), meta)