import os
import math
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim


from lib.Network_Multimodal_LogitsGatedFusion import Multi_Network
from utils.data_val_Multi import get_loader_Multi, test_dataset_Multi
from utils.utils import clip_gradient



print("CUDA Available:", torch.cuda.is_available())


def set_seed(seed=34):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, opt):
    global step
    model.train()
    loss_all = 0.0
    epoch_step = 0

    bce_pix = torch.nn.BCEWithLogitsLoss(reduction='none')

    try:
        for i, (RGB, T, gts, _) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            RGB = RGB.cuda(device=device_ids[0])
            T = T.cuda(device=device_ids[0])
            gts = gts.cuda(device=device_ids[0])

            preds = model(RGB, T)
            logit_r, logit_t, logit_f, w_full = preds[4], preds[13], preds[18], preds[19]

            loss_fusion = structure_loss(logit_f, gts)
            loss = loss_fusion


            R_rgb_raw = None
            R_t_raw   = None
            R_rgb = None
            R_t   = None

            if opt.use_da_ad:
                if epoch == 1:
                    print(
                        "===> [DaAD ENABLED] "
                        f"use_da_ad_loss={opt.use_da_ad_loss}"
                    )

                L_full = bce_pix(logit_f, gts)

                L_no_r = bce_pix(torch.clamp(w_full[:, 1:2], min=0.3) * logit_t, gts)
                L_no_t = bce_pix(torch.clamp(w_full[:, 0:1], min=0.3) * logit_r, gts)

                with torch.no_grad():
                    d_rgb_map = (L_no_r - L_full).clamp_min(0.0)
                    d_t_map   = (L_no_t - L_full).clamp_min(0.0)
                    denom = d_rgb_map + d_t_map + 1e-6

                    R_rgb_raw = d_rgb_map / denom
                    R_t_raw   = d_t_map   / denom

                    gap  = (L_no_r - L_no_t).abs()
                    base = (L_no_r + L_no_t) * 0.5 + 1e-6
                    diff_coef = (gap / base).clamp(0.0, 1.0)

                    common = 1.0 - diff_coef

                    R_rgb = common * 1.0 + diff_coef * R_rgb_raw
                    R_t   = common * 1.0 + diff_coef * R_t_raw

                if opt.use_da_ad_loss:
                    bce_rgb_branch = bce_pix(logit_r, gts)
                    bce_t_branch   = bce_pix(logit_t, gts)

                    bce_rgb_weighted = (bce_rgb_branch * R_rgb).sum((2, 3))
                    rgb_norm = R_rgb.sum((2, 3)) + 1e-8
                    L_rgb_pix = (bce_rgb_weighted / rgb_norm).mean()

                    bce_t_weighted = (bce_t_branch * R_t).sum((2, 3))
                    t_norm = R_t.sum((2, 3)) + 1e-8
                    L_t_pix = (bce_t_weighted / t_norm).mean()

                    loss = loss + opt.lambda_da_ad * (L_rgb_pix + L_t_pix)

            if R_rgb is None:
                with torch.no_grad():
                    L_full = bce_pix(logit_f, gts)

                    L_no_r = bce_pix(torch.clamp(w_full[:, 1:2], min=0.3) * logit_t, gts)
                    L_no_t = bce_pix(torch.clamp(w_full[:, 0:1], min=0.3) * logit_r, gts)

                    d_rgb_map = (L_no_r - L_full).clamp_min(0.0)
                    d_t_map   = (L_no_t - L_full).clamp_min(0.0)

                    denom = d_rgb_map + d_t_map + 1e-6
                    R_rgb_raw = d_rgb_map / denom
                    R_t_raw   = d_t_map   / denom

                    R_rgb = R_rgb_raw
                    R_t   = R_t_raw

            c_rgb_inst = float(R_rgb_raw.mean().detach().cpu().item())
            c_t_inst   = float(R_t_raw.mean().detach().cpu().item())



            loss.backward()

            suppression_status = 'Balanced' if opt.use_igm else None
            if opt.use_igm and epoch > opt.igm_warmup_epochs:
                with torch.no_grad():
                    delta_c = (c_rgb_inst - c_t_inst)
                    delta_c_val = float(delta_c)
                logging.info(f"[IGM] delta_c (c_rgb_inst - c_t_inst) = {delta_c_val:+.6f}")
                with torch.no_grad():
                    conf_mean = (w_full[:, 0] - w_full[:, 1]).abs().mean().detach()
                    conf_gate = torch.sigmoid((conf_mean - opt.igm_conf_deadband) * opt.igm_conf_gain)
                    imb = abs(c_rgb_inst - c_t_inst)
                    imb_gate = math.tanh(imb / (opt.igm_imb_scale + 1e-6))
                    intensity = opt.igm_alpha * float(conf_gate) * float(imb_gate)

                    coeff_rgb_val = 1.0
                    coeff_t_val = 1.0
                    if (c_rgb_inst - c_t_inst) > opt.igm_deadband:
                        coeff_rgb_val = max(opt.igm_coeff_min, 1.0 - intensity)
                    elif (c_t_inst - c_rgb_inst) > opt.igm_deadband:
                        coeff_t_val = max(opt.igm_coeff_min, 1.0 - intensity)

                    coeff_rgb = coeff_rgb_val
                    coeff_t = coeff_t_val

                    if coeff_rgb_val < 1.0:
                        suppression_status = f'Suppress RGB ×{coeff_rgb_val:.4f}'
                    elif coeff_t_val < 1.0:
                        suppression_status = f'Suppress T ×{coeff_t_val:.4f}'

                rem11_rgb_pre_vals, rem11_t_pre_vals = [], []
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if 'REM11_RGB' in name and 'weight' in name:
                        rem11_rgb_pre_vals.append(p.grad.detach().abs().mean())
                    elif 'REM11_T' in name and 'weight' in name:
                        rem11_t_pre_vals.append(p.grad.detach().abs().mean())
                pre_rgb_mean = float(torch.stack(rem11_rgb_pre_vals).mean().cpu().item()) if rem11_rgb_pre_vals else 0.0
                pre_t_mean   = float(torch.stack(rem11_t_pre_vals).mean().cpu().item())   if rem11_t_pre_vals   else 0.0

                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if 'REM11_RGB' in name and 'weight' in name:
                        p.grad.mul_(coeff_rgb)
                    elif 'REM11_T' in name and 'weight' in name:
                        p.grad.mul_(coeff_t)

                rem11_rgb_post_vals, rem11_t_post_vals = [], []
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if 'REM11_RGB' in name and 'weight' in name:
                        rem11_rgb_post_vals.append(p.grad.detach().abs().mean())
                    elif 'REM11_T' in name and 'weight' in name:
                        rem11_t_post_vals.append(p.grad.detach().abs().mean())
                post_rgb_mean = float(torch.stack(rem11_rgb_post_vals).mean().cpu().item()) if rem11_rgb_post_vals else 0.0
                post_t_mean   = float(torch.stack(rem11_t_post_vals).mean().cpu().item())   if rem11_t_post_vals   else 0.0

                logging.info(
                    "[IGM] REM11_RGB gradient mean | pre: {:.6e} -> post: {:.6e} (x{:.3f});  "
                    "REM11_T mean | pre: {:.6e} -> post: {:.6e} (x{:.3f})".format(
                        pre_rgb_mean, post_rgb_mean, (post_rgb_mean / (pre_rgb_mean + 1e-12)),
                        pre_t_mean,   post_t_mean,   (post_t_mean   / (pre_t_mean   + 1e-12))
                    )
                )

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 5 == 0 or i == total_step or i == 1:
                progress = '[Epoch {:03d}/{:03d} | Step {:04d}/{:04d}] Loss {:.4f}'.format(
                    epoch, opt.epoch, i, total_step, loss.data)
                if suppression_status is not None:
                    progress += f' | {suppression_status}'
                print(progress)
                logging.info('[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.format(
                    epoch, opt.epoch, i, total_step, loss.data))

        loss_all /= max(1, epoch_step)
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))

        if epoch > 49 and epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'Net_epoch_{epoch}.pth'))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, f'Net_epoch_{epoch + 1}.pth'))
        print('Save checkpoints successfully!')
        raise



def val(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        iou_sum = 0
        for i in range(test_loader.size):
            RGB, Thermal, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            RGB = RGB.cuda(device=device_ids[0])
            Thermal = Thermal.cuda(device=device_ids[0])
            
            result = model(RGB, Thermal)
            res = F.interpolate(result[18], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            pred_bin = (res > 0.5).astype(np.float32)
            gt_bin = (gt > 0.5).astype(np.float32)
            intersection = np.sum(pred_bin * gt_bin)
            union = np.sum(pred_bin) + np.sum(gt_bin) - intersection + 1e-8
            iou = intersection / union
            iou_sum += iou

            
        mae = mae_sum / test_loader.size
        avg_iou = iou_sum / test_loader.size



        print('Epoch: {}, MAE: {:.4f}, IoU: {:.4f}, bestMAE: {:.4f}, bestEpoch: {}.'
              .format(epoch, mae, avg_iou, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
            best_epoch = 1
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_best.pth'))
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info('[Val Info]:Epoch:{} MAE:{} IoU:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae,avg_iou, best_epoch, best_mae))
    return mae
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--trainsize', type=int, default=384)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, default='0,1')

    parser.add_argument('--igm_warmup_epochs', type=int, default=0)

    parser.add_argument('--train_root', type=str, required=True)
    parser.add_argument('--val_root', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)


    parser.add_argument('--use_da_ad', action='store_true',
        help='Compute DaAD contribution maps and weights')
    parser.add_argument('--use_da_ad_loss', action='store_true',
        help='Enable the contribution-weighted DaAD loss')
    parser.add_argument('--lambda_da_ad', type=float, default=1.0,
        help='Weight for the DaAD branch loss')


    parser.add_argument('--use_igm', action='store_true',
        help='Enable IGM gradient modulation')

    parser.add_argument('--igm_alpha', type=float, default=0.1)
    parser.add_argument('--igm_deadband', type=float, default=0.02)
    parser.add_argument('--igm_coeff_min', type=float, default=0.5)

    parser.add_argument('--igm_conf_deadband', type=float, default=0.03)
    parser.add_argument('--igm_conf_gain', type=float, default=8.0)
    parser.add_argument('--igm_imb_scale', type=float, default=0.15)

    opt = parser.parse_args()
    set_seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    print('USE GPU', opt.gpu_id)

    global device_ids, total_step, best_mae, best_epoch, step
    device_ids = [0,1]
    model = torch.nn.DataParallel(Multi_Network(channels=96), device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)


    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)

    print('load data...')
    train_loader = get_loader_Multi(opt.train_root + 'RGB/',
                                    opt.train_root + 'T/',
                                    opt.train_root + 'GT/',
                                    opt.train_root + 'Edge/',
                                    opt.batchsize, opt.trainsize, num_workers=8)
    val_loader = test_dataset_Multi(opt.val_root + 'RGB/',
                                    opt.val_root + 'T/',
                                    opt.val_root + 'GT/',
                                    opt.trainsize)
    total_step = len(train_loader)
    best_mae = 1e9
    best_epoch = 0
    step = 0

    logging.basicConfig(filename=os.path.join(save_path, 'log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Multi-Network-Train")
    logging.info(
        f'Config: epoch={opt.epoch}; lr={opt.lr}; batchsize={opt.batchsize}; trainsize={opt.trainsize}; '
        f'clip={opt.clip}; load={opt.load}; save_path={save_path}; '
        f'use_da_ad={opt.use_da_ad}; use_da_ad_loss={opt.use_da_ad_loss}; lambda_da_ad={opt.lambda_da_ad}; '
        f'use_igm={opt.use_igm}; igm_alpha={opt.igm_alpha}; '
        f'igm_warmup_epochs={opt.igm_warmup_epochs}; igm_deadband={opt.igm_deadband}; '
        f'igm_coeff_min={opt.igm_coeff_min}; igm_conf_deadband={opt.igm_conf_deadband}; '
        f'igm_conf_gain={opt.igm_conf_gain}; igm_imb_scale={opt.igm_imb_scale}; '
    )



    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=opt.epoch, eta_min=1e-5)
    print("Start train...")

    for epoch in range(1, opt.epoch + 1):

        train(train_loader, model, optimizer, epoch, save_path, opt)
        mae = val(val_loader, model, epoch, save_path)

        scheduler.step()

        torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_last.pth'))


if __name__ == "__main__":
    main()
