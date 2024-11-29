# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
#%%
import math
import sys
from typing import Iterable, Optional
from PIL import Image
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torchvision.transforms as T
import os.path as osp
import os
from losses import DistillationLoss
import cv2
import numpy as np
import utils
import torch.nn.functional as F

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
from col_to_alpha import color_to_alpha


v = 1
#colors = [white,빨강(채도50%),빨강(채도100%)]
# rgb로 변경 -> mcl.hsv_to_rgb(h,s,v)


# cmap = LinearSegmentedColormap.from_list('my_cmap',color,gamma=1,N=255)
def imtoclr2(img, L, name):
    coef = int(256/L)
    colors1 = np.array([mcl.hsv_to_rgb((25/360,0.3,v)),mcl.hsv_to_rgb((25/360,0.8,v)),mcl.hsv_to_rgb((25/360,1,v))])
    colors2 = np.array([mcl.hsv_to_rgb((252/360,0.3,v)),mcl.hsv_to_rgb((252/360,0.8,v)),mcl.hsv_to_rgb((252/360,1,v))])
    colors3 = np.array([mcl.hsv_to_rgb((139/360,0.3,v)),mcl.hsv_to_rgb((139/360,0.8,v)),mcl.hsv_to_rgb((139/360,1,v))])
    colors4 = np.array([mcl.hsv_to_rgb((296/360,0.3,v)),mcl.hsv_to_rgb((296/360,0.8,v)),mcl.hsv_to_rgb((296/360,1,v))])
    colors1 = LinearSegmentedColormap.from_list('my_cmap',colors1,gamma=1,N=256)(np.linspace(0., 1, 64))
    colors2 = LinearSegmentedColormap.from_list('my_cmap',colors2,gamma=1,N=256)(np.linspace(0., 1, 64))
    colors3 = LinearSegmentedColormap.from_list('my_cmap',colors3,gamma=1,N=256)(np.linspace(0., 1, 64))
    colors4 = LinearSegmentedColormap.from_list('my_cmap',colors4,gamma=1,N=256)(np.linspace(0., 1, 64))
    color = np.vstack([colors1,colors2, colors3, colors4])
    cmap = LinearSegmentedColormap.from_list('my_cmap',color,gamma=1,N=256)
    img = ((img+1)*coef)-1
    
    plt.imsave(name, img, cmap=cmap)
    # for i in range(L):
    # #     # gamma = i*coef
    #     tempname = "_{}.jpg".format(i)
    #     sname = name + tempname
        # temp = g_img*(g_img<coef)*(g_img>=0) + whites*(g_img>=coef)+whites*(g_img<0)
        # temp = color_to_alpha(temp, (255, 0, 255), 0.5 * 18, 0.75 * 193, 'cube', 'smooth')
        # temp = Image.fromarray(np.ubyte(new_pixels))
        
def imtoclr(img, L, name):
    coef = int(256/L)
    colors1 = np.array([mcl.hsv_to_rgb((25/360,0.5,v)),mcl.hsv_to_rgb((25/360,0.8,v)),mcl.hsv_to_rgb((25/360,1,v))])
    colors2 = np.array([mcl.hsv_to_rgb((252/360,0.5,v)),mcl.hsv_to_rgb((252/360,0.8,v)),mcl.hsv_to_rgb((252/360,1,v))])
    colors3 = np.array([mcl.hsv_to_rgb((139/360,0.5,v)),mcl.hsv_to_rgb((139/360,0.8,v)),mcl.hsv_to_rgb((139/360,1,v))])
    colors4 = np.array([mcl.hsv_to_rgb((296/360,0.5,v)),mcl.hsv_to_rgb((296/360,0.8,v)),mcl.hsv_to_rgb((296/360,1,v))])
    colors1 = LinearSegmentedColormap.from_list('my_cmap',colors1,gamma=1,N=256)(np.linspace(0., 1, coef))
    colors2 = LinearSegmentedColormap.from_list('my_cmap',colors2,gamma=1,N=256)(np.linspace(0., 1, coef))
    colors3 = LinearSegmentedColormap.from_list('my_cmap',colors3,gamma=1,N=256)(np.linspace(0., 1, coef))
    colors4 = LinearSegmentedColormap.from_list('my_cmap',colors4,gamma=1,N=256)(np.linspace(0., 1, coef))
    white = np.array([1,1,1, 0])
    colors1 = np.vstack((colors1, white))
    colors2 = np.vstack((colors2, white))
    colors3 = np.vstack((colors3, white))
    colors4 = np.vstack((colors4, white))
    colors1 = LinearSegmentedColormap.from_list('my_cmap',colors1,gamma=1,N=coef+1)
    colors2 = LinearSegmentedColormap.from_list('my_cmap',colors2,gamma=1,N=coef+1)
    colors3 = LinearSegmentedColormap.from_list('my_cmap',colors3,gamma=1,N=coef+1)
    colors4 = LinearSegmentedColormap.from_list('my_cmap',colors4,gamma=1,N=coef+1)
    img = (img+1)*coef-1
    whites = np.ones_like(img)*(coef)
    for i in range(L):
        gamma = i*coef
        g_img = img-gamma
        if gamma < 64:
            cmap = colors1
        elif gamma >=64 and gamma<128:
            cmap = colors2
        elif gamma >=128 and gamma<192:
            cmap = colors3
        else:
            cmap = colors4
        tempname = "_{}.jpg".format(i)
        sname = name + tempname
        temp = g_img*(g_img<coef)*(g_img>=0) + whites*(g_img>=coef)+whites*(g_img<0)
        # temp = color_to_alpha(temp, (255, 0, 255), 0.5 * 18, 0.75 * 193, 'cube', 'smooth')
        # temp = Image.fromarray(np.ubyte(new_pixels))
        plt.imsave(sname, temp, cmap=cmap)
    
    
    # cmap


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs, _, col_loss, hyp_loss = model(samples)
            # outputs = x[:,0]
            # vismap = x[:,1:]
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
                print("cls loss : {}".format(loss))
                # loss += sum(hyp_loss[1])
                loss += hyp_loss[1]
                loss += col_loss
                # loss += 1 - torch.mean(F.cosine_similarity(outputs, x_ori))
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    os.makedirs("/home/diml/khj/deit/eff_visualize", exist_ok = True)
    img_pth = "/home/diml/khj/deit/eff_visualize/epoch_{}".format(epoch)
    inv_normalize = T.Normalize(mean= [-m/s for m, s in zip(mean, std)], std= [1/s for s in std])
    Re = T.Resize(224, interpolation=0)
    os.makedirs(img_pth, exist_ok = True)

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    cnt = 0
    # switch to evaluation mode
    model.eval()
    model.requires_grad = False
    count = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        ori = inv_normalize(images[0]).permute(1, 2, 0).cpu().numpy()
        ori_name = osp.join(img_pth, "{}_0.jpg".format(cnt))
        ori = (ori/ori.max())*255
        ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
        cv2.imwrite(ori_name, ori/ori.max()*255)
        # compute output
        # if count == 0:
        #     flops = FlopCountAnalysis(model, images[0].unsqueeze(0))
        #     print(flop_count_table(flops))
        #     count += 1
        with torch.cuda.amp.autocast():
            output, feat, node_memory, out_hyp = model(images)
            loss = criterion(output, target)
            for idx, node in enumerate(node_memory):
                temp_node = 0
                if idx > 0:
                    for i in range(idx):
                        temp_node += node[:,i::2**(idx+0)]
                    temp_node = temp_node/(2**(idx+0))
                else: 
                    temp_node = node
                if idx < len(node_memory):
                    B, L, C = temp_node.shape
                    feat = F.normalize(feat, dim=-1)
                    temp_node = F.normalize(temp_node, dim=-1)
                    vis_data = torch.einsum("bnc, bmc->bnm",feat, temp_node)
                    hyp_data = torch.einsum("bnc, bmc->bnm",feat, out_hyp[:,idx])
                    _, nodemap= torch.max(vis_data, -1)
                    # actmap, _ = torch.max(hyp_data )
                    vis_fig = nodemap[0].view(1, 14, 14)
                    hyp_fig = hyp_data[0].squeeze(-1).view(1, 14, 14)
                    vis_fig = Re(vis_fig).squeeze(0).detach().cpu().numpy()
                    hyp_fig = Re(hyp_fig).squeeze(0).detach().cpu().numpy()
                    hyp_fig = np.uint8((hyp_fig/hyp_fig.max())*255)
                    hyp_fig = cv2.applyColorMap(hyp_fig, cv2.COLORMAP_JET)
                    imgname = osp.join(img_pth, "{}_{}.jpg".format(cnt, 2**(4-idx)))
                    himgname = osp.join(img_pth, "{}_{}_hyp.jpg".format(cnt, 2**(4-idx)))
                    imtoclr2(vis_fig, L, imgname)
                    # cv2.imwrite(imgname, vis_fig)
                    # cv2.imwrite(himgname, hyp_fig)
                else:
                    pass
        cnt += 1
        

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
