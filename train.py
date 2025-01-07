
import os
import time
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from scripts import models
from scripts.dataset import MetricDataset, Inshop_Dataset, BalancedBatchSampler, RGBToBGR
from scripts.evaluate import evaluation
from tensorboardX import SummaryWriter

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from config import cfg, cfg_from_file


def fix_random_seeds(seed=2024):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')
    parser.add_argument('--alpha', type=float, default=32, help='alpha value')
    parser.add_argument('--mrg', type=float, default=0.1, help='eta value')
    parser.add_argument('--partial_rate', type=float, default=1.0, help='partial rate')
    parser.add_argument('--imbalance', action='store_true', help='imbalance setting')
    parser.add_argument('--gamma', type=int, default=0, help='imbalance rate')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    return parser.parse_args()


def trainer(cfg, opt):
    cfg_path = opt.cfg_file
    device = "cuda"
    # data prepare
    model_path = cfg.TRAIN.MODEL_PATH
    data_path = cfg.TRAIN.DATASET
    epoches = cfg.TRAIN.EPOCHES
    if data_path != 'data/IN_SHOP/class/train':
        mdata = MetricDataset(cfg.TRAIN.DATASET, cfg.MODEL.BACKBONE, scale=cfg.TRAIN.SCALE)
    else:
        if cfg.MODEL.BACKBONE == "BNInception":
                inception_mean = [0.4078, 0.4588, 0.502]
                inception_std = [0.0039, 0.0039, 0.0039]
                mdata = Inshop_Dataset(
                root = 'data',
                mode = 'train',
                transform = transforms.Compose([
                             RGBToBGR(),
                            transforms.RandomResizedCrop((224, 224), cfg.TRAIN.SCALE),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=inception_mean, std=inception_std)
                        ])
            ) 
        else:
            mdata = Inshop_Dataset(
                root = 'data',
                mode = 'train',
                transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
            )
    
    n_classes = cfg.TRAIN.NUM_SAMPLED_CLASSES
    n_per_class = cfg.TRAIN.NUM_PER_CLASS
    batch_size = n_classes * n_per_class
    sampler = BalancedBatchSampler(mdata.targets, n_classes=n_classes, n_samples=n_per_class, partial_rate=opt.partial_rate, imbalance=opt.imbalance, gamma=opt.gamma)
    train_loader = torch.utils.data.DataLoader(
        mdata,  
        batch_sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
    )

    # tensorboard init
    save_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print("expriment running at:", save_time)
    tensorboard_path = os.path.join(cfg.TRAIN.BOARD_LOG_PATH, opt.cfg_file.split('/')[-1].split('.')[0], save_time)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    shutil.copyfile(cfg_path, os.path.join(tensorboard_path, cfg_path.split('/')[-1]))
    try:
        summary_writer = SummaryWriter(tensorboard_path, comment="esdml")
        print('log tensorboard summaty at {}'.format(tensorboard_path))
    except:
        summary_writer = None
        print('Can not use tensorboard')

    # model init
    embedding_size = cfg.MODEL.EMBEDDING_SIZE
    pretrained = cfg.MODEL.PRETRAINED
    is_norm = cfg.MODEL.NORM
    is_pooling=cfg.MODEL.POOLING
    bn_freeze = cfg.SOLVER.BN_FREEZE
    patch_freeze = cfg.SOLVER.PATCH_FREEZE
    print('pool', is_pooling)
    model = models.Metric_Model(cfg, embedding_size=embedding_size, n_class=len(mdata.classes), pretrained=pretrained, is_norm=is_norm, is_pooling=is_pooling, bn_freeze=bn_freeze, alpha=opt.alpha, mrg=opt.mrg).to(device)

    # optim init
    lr_metric = cfg.SOLVER.BASE_LR
    metric_decay_step = cfg.SOLVER.METRIC_STEPS
    metirc_gamma = cfg.SOLVER.METRIC_GAMMA

    if cfg.SOLVER.METRIC_TYPE == 'Adam': 
        optimizer_general = optim.Adam
    elif cfg.SOLVER.METRIC_TYPE == 'AdamW':
        optimizer_general = optim.AdamW
    elif cfg.SOLVER.METRIC_TYPE == 'rmsprop':
        optimizer_general = optim.RMSprop
    else:
        raise NotImplementedError
    
    metric_model_param = [{'params':model.feature_extractor.backbone.parameters(), 'lr':lr_metric}, 
                          {'params':model.feature_extractor.fc.parameters(), 'lr': lr_metric * cfg.SOLVER.EMBEDDING_RATIO}]
    if cfg.SOLVER.METRIC_TYPE == 'Adam':
        optimizer_c = optimizer_general(metric_model_param, lr=lr_metric)
    elif cfg.SOLVER.METRIC_TYPE == 'AdamW':
        optimizer_c = optimizer_general(metric_model_param, lr=lr_metric, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.METRIC_TYPE == 'rmsprop':
        optimizer_c = optimizer_general(metric_model_param, lr=lr_metric, alpha=0.9, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=0.9)

    if cfg.SOLVER.METRIC_TYPE == 'AdamW':
        optimizer_pa = optimizer_general(model.proxies.parameters(), lr=lr_metric*cfg.LOSS.PROXY_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.GEN_TYPE == 'rmsprop':
        optimizer_pa = optimizer_general(model.proxies.parameters(), lr=lr_metric*cfg.LOSS.PROXY_LR, alpha=0.9, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=0.9)
    elif cfg.SOLVER.METRIC_TYPE == 'Adam':
        optimizer_pa = optimizer_general(model.proxies.parameters(), lr=lr_metric*cfg.LOSS.PROXY_LR)

    if cfg.SOLVER.LR_POLICY == 'step':
        scheduler_pa = optim.lr_scheduler.StepLR(optimizer_pa, step_size=metric_decay_step, gamma=metirc_gamma)
    elif cfg.SOLVER.LR_POLICY == 'multistep':
        scheduler_pa = optim.lr_scheduler.MultiStepLR(optimizer_pa, milestones=metric_decay_step, gamma=metirc_gamma)
    elif cfg.SOLVER.LR_POLICY == 'cos':
        scheduler_pa = optim.lr_scheduler.CosineAnnealingLR(optimizer_pa, cfg.SOLVER.METRIC_STEPS, eta_min=0, last_epoch=-1)
    elif cfg.SOLVER.LR_POLICY == 'exp':
        scheduler_pa = optim.lr_scheduler.ExponentialLR(optimizer_pa, gamma=0.9)
    else:
        scheduler_pa = None

    if cfg.SOLVER.LR_POLICY == 'step':
        scheduler_c = optim.lr_scheduler.StepLR(optimizer_c, step_size=metric_decay_step, gamma=metirc_gamma)
    elif cfg.SOLVER.LR_POLICY == 'multistep':
        scheduler_c = optim.lr_scheduler.MultiStepLR(optimizer_c, milestones=metric_decay_step, gamma=metirc_gamma)
    elif cfg.SOLVER.LR_POLICY == 'cos':
        scheduler_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, float(epoches))
    elif cfg.SOLVER.LR_POLICY == 'exp':
        scheduler_c = optim.lr_scheduler.ExponentialLR(optimizer_c, gamma=0.9)
    else:
        scheduler_c = None

    # start training
    recalls_dict = {}
    map_dict = {}
    Rp_dict = {}
    best_result = {}
    best_r1 = 0
    iters_per_epoch = int(len(mdata)*opt.partial_rate)//(batch_size)

    for epoch in range(epoches):
        model.train()
        if bn_freeze:
            modules = model.feature_extractor.modules()
            for m in modules: 
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        if patch_freeze:
            modules = model.feature_extractor.named_modules()
            for name, m in modules:
                if 'patch_embed' in name or 'pos_drop' in name:
                    m.eval()

        if cfg.SOLVER.WARM_UP_EPOCH > 0:
            freeze_model_param = set(model.feature_extractor.parameters()).difference(set(model.feature_extractor.fc.parameters()))
            if epoch < cfg.SOLVER.WARM_UP_EPOCH:
                for param in list(freeze_model_param):
                    param.requires_grad = False
            else:
               for param in list(freeze_model_param):
                    param.requires_grad = True

        with tqdm(total=iters_per_epoch, desc=f'Epoch {epoch + 1}/{epoches}', ncols=130, postfix=dict, mininterval=0.3) as pbar:
            for i, batch in enumerate(train_loader):
                x_batch, label = batch

                jm, metrics = model(x_batch.to(device), label.to(device), opt_c=optimizer_c, opt_pa=optimizer_pa) 

                pbar.set_postfix(**{"Jm": jm.item()})
                pbar.update(1)
                summary_writer.add_scalars('hsg/losses', metrics, epoch * (len(mdata) // batch_size) + i)           

        if (epoch+1) % cfg.TEST.EVAL_ITER == 0:
            if not os.path.exists(os.path.join(model_path, opt.cfg_file.split('/')[-1].split('.')[0], save_time)):
                os.makedirs(os.path.join(model_path, opt.cfg_file.split('/')[-1].split('.')[0], save_time))

            if cfg.TEST.DATASET == 'data/SOP/class/test':
                nmi, f1, recalls, map_R, R_p = evaluation(cfg, model)
                for i, k in enumerate([1, 10, 100, 1000]):
                    recalls_dict.update({str(k): round(float(recalls[i]),3)})
            elif cfg.TEST.DATASET == 'data/IN_SHOP/class/test':
                recalls, map_R, R_p  = evaluation(cfg, model)
                for i, k in enumerate([1, 10, 20, 30, 40, 50]):
                    recalls_dict.update({str(k): round(float(recalls[i]),3)})
            else:
                nmi, f1, recalls, map_R, R_p = evaluation(cfg, model)
                for i, k in enumerate([1, 2]):
                    recalls_dict.update({str(k): round(float(recalls[i]),3)})

            map_dict.update({"map_R": round(map_R, 3)})
            Rp_dict.update({"R_p": round(R_p, 3)})
            
            summary_writer.add_scalars('hsg/recalls', recalls_dict, 10 * epoch * (len(mdata) // batch_size) + i)
            summary_writer.add_scalars('hsg/map_R', map_dict, 10 * epoch * (len(mdata) // batch_size) + i)
            summary_writer.add_scalars('hsg/R_p', Rp_dict, 10 * epoch * (len(mdata) // batch_size) + i)

            if recalls[0] > best_r1:
                best_r1 = recalls[0]
                best_result['recall'] = recalls_dict.copy()
                best_result['map_R'] = map_dict.copy()
                best_result['R_p'] = Rp_dict.copy()
                torch.save(model.state_dict(), os.path.join(model_path, opt.cfg_file.split('/')[-1].split('.')[0], save_time, 'best_R1.pth'))
            
        if scheduler_c is not None:
            scheduler_c.step()
        if scheduler_pa is not None:
            scheduler_pa.step()
        
    os.makedirs('logs/', exist_ok=True)
    print('save log for: ' + save_time + '\n')
    with open('logs/' + opt.cfg_file.split('/')[-1][:-4] + '_log.txt', 'a') as f:
        f.write(opt.cfg_file.split('/')[-1] + '\n' +  save_time + '\t' + str(best_result) + '\n')
        f.write('seed:'+ str(opt.seed) + '\t' + 'alpha:' + str(opt.alpha) + '\t' + 'mrg:' + str(opt.mrg) + '\n' \
                + 'partial:' + str(opt.partial_rate) + '\n' + 'imbalance:' + str(opt.imbalance) + '\t' + 'gamma:' + str(opt.gamma) + '\n')
        f.write('_'*50 + '\n')

if __name__ == "__main__":
    opt = parse_args()
    cfg_from_file(opt.cfg_file)
    fix_random_seeds(opt.seed)
    trainer(cfg, opt)
