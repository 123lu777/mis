import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True   # 免费的 TF32 加速（4090/Ampere 架构）

import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data.data_RGB import get_training_data, get_validation_data

# =============================================
# 导入带物理先验约束的 HAHA 版本模型
# =============================================
from MISCFilterNet_HAHA import MISCKernelNet_HAHA as myNet

from loss import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tools.get_parameter_number import get_parameter_number
import kornia

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
start_epoch = 1


# =============================================
# 所有参数在这里直接配置，右键即可运行
# =============================================
class Config:
    # 数据集路径
    train_dir = './dataset/GOPRO_Large'
    train_meta = './dataset/GOPRO_Large/GOPRO_train_list.txt'
    val_dir = './dataset/GOPRO_Large'
    val_meta = './dataset/GOPRO_Large/GOPRO_test_list.txt'

    # 模型保存路径
    model_save_dir = './checkpoints_haha'
    dataset = 'GoPro'
    session = 'MISCFilter_HAHA_GoPro'

    # 训练参数
    patch_size = 256  # [GoPro, HIDE, RealBlur]=256, [DPDD]=512
    num_epochs = 6000
    # 双卡 GPU + AMP FP16：batch_size=12（每卡 6 张），介于 8 与 16 之间，4 的倍数保证 GPU 对齐，GPU 利用率优于 8
    batch_size = 12
    val_epochs = 10
    print_epochs = 2

    # 可变形卷积设置
    use_deform_in_feat = True
    use_deform_in_encoder = True

    # =============================================
    # 物理先验约束权重（新增）
    # =============================================
    rigid_smooth_weight = 0.05    # 刚性运动平滑约束权重：L_smooth = Σ sqrt(|∇u|²+ε²) + sqrt(|∇v|²+ε²)
    rotation_aware_weight = 0.03  # 无中心旋转感知约束权重：L_rotation = mean(clamp(div - curl, min=0))
    flow_gradient_weight = 0.02   # 流梯度二阶平滑约束权重：L_gradient = Σ sqrt(|∇²u|²+ε²) + sqrt(|∇²v|²+ε²)

    # 恢复训练设置
    RESUME = True  # 首次训练设为 False，恢复训练设为 True
    Pretrain = False
    model_pre_dir = ''


args = Config()

dataset = args.dataset
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, dataset, session)
utils.mkdir(model_dir)
log_dir = os.path.join(args.model_save_dir, dataset, session, 'log.txt')

train_dir = args.train_dir
val_dir = args.val_dir

train_meta = args.train_meta
val_meta = args.val_meta

num_epochs = args.num_epochs
batch_size = args.batch_size
val_epochs = args.val_epochs

start_lr = 3e-4  # 线性缩放规则：原始 batch_size=16 对应 4e-4，batch_size=12 → 4e-4×(12/16)=3e-4
end_lr = 1e-6

######### Model ###########
model_restoration = myNet(
    use_deform_in_feat=args.use_deform_in_feat,
    use_deform_in_encoder=args.use_deform_in_encoder,
    rigid_smooth_weight=args.rigid_smooth_weight,
    rotation_aware_weight=args.rotation_aware_weight,
    flow_gradient_weight=args.flow_gradient_weight,
)

# print number of model
total_num, trainable_num = get_parameter_number(model_restoration)
print('=' * 60)
print('Model: MISCKernelNet_HAHA (Deformable + Physical Prior Constraints)')
print('Use Deform in Feature Extraction:', args.use_deform_in_feat)
print('Use Deform in Encoder/Decoder:', args.use_deform_in_encoder)
print('Rigid Smooth Weight:', args.rigid_smooth_weight)
print('Rotation Aware Weight:', args.rotation_aware_weight)
print('Flow Gradient Weight:', args.flow_gradient_weight)
print('=' * 60)
print('Total:  ', total_num)
print('Trainable: ', trainable_num)
with open(log_dir, "a+") as f:
    f.write('=' * 60 + '\n')
    f.write('Model: MISCKernelNet_HAHA (Deformable + Physical Prior Constraints)\n')
    f.write('Use Deform in Feature Extraction: {}\n'.format(args.use_deform_in_feat))
    f.write('Use Deform in Encoder/Decoder: {}\n'.format(args.use_deform_in_encoder))
    f.write('Rigid Smooth Weight: {}\n'.format(args.rigid_smooth_weight))
    f.write('Rotation Aware Weight: {}\n'.format(args.rotation_aware_weight))
    f.write('Flow Gradient Weight: {}\n'.format(args.flow_gradient_weight))
    f.write('=' * 60 + '\n')
    f.write('Total: {}\n'.format(total_num))
    f.write('Trainable: {}\n'.format(trainable_num))

model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
# 延长 warmup 至 10 个 epoch：LR 从 start_lr/10 逐步线性升至 start_lr，
# 避免前几个 epoch 因 LR 跳变过快导致 FP16 梯度溢出（NaN loss）
warmup_epochs = 10
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

RESUME = args.RESUME
Pretrain = args.Pretrain
model_pre_dir = args.model_pre_dir

######### Pretrain ###########
if Pretrain:
    utils.load_checkpoint(model_restoration, model_pre_dir)
    print('------------------------------------------------------------------------------')
    print("==> Retrain Training with:  " + model_pre_dir)
    print('------------------------------------------------------------------------------')

######### Resume ###########
if RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

# AMP GradScaler：FP16 混合精度训练，在 4090 Tensor Core 上提速约 1.5-2×
scaler = GradScaler('cuda')

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, train_meta, {'patch_size': patch_size})
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=8, drop_last=True, pin_memory=True, prefetch_factor=4)

val_dataset = get_validation_data(val_dir, val_meta, {'patch_size': patch_size})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False,
                        num_workers=8, drop_last=False, pin_memory=True, prefetch_factor=4)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
print('===> Loading datasets')
with open(log_dir, "a+") as f:
    f.write('===> Start Epoch {} End Epoch {} \n'.format(start_epoch, num_epochs + 1))
    f.write('===> Loading datasets\n')

best_psnr = 0
best_epoch = 0
iter = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(train_loader, 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target_ = data[0].cuda()
        input_ = data[1].cuda()
        target = kornia.geometry.transform.build_pyramid(target_, 3)

        # AMP 自动混合精度前向 + 损失计算
        with autocast('cuda'):
            restored, restored_inter, kernal_loss = model_restoration(input_)

            loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(
                restored[2], target[2])
            loss_char = criterion_char(restored[0], target[0]) + criterion_char(restored[1], target[1]) + criterion_char(
                restored[2], target[2])
            loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(restored[1], target[1]) + criterion_edge(
                restored[2], target[2])
            loss_char_inter = criterion_char(restored_inter[0], target[0]) + criterion_char(restored_inter[1],
                                                                                            target[1]) + criterion_char(
                restored_inter[2], target[2])

            loss = loss_char + loss_char_inter + 0.01 * loss_fft + 0.05 * loss_edge + kernal_loss.mean()

        # NaN/Inf 守卫：跳过损坏的 batch，防止 NaN 污染优化器状态
        if not torch.isfinite(loss):
            print(f'[WARN] epoch {epoch} iter {i}: loss={loss.item()}, skipping batch')
            for param in model_restoration.parameters():
                param.grad = None
            continue

        scaler.scale(loss).backward()
        # 梯度裁剪（unscale 后再 clip）：防止 FP16 混合精度下梯度爆炸引发 NaN
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=0.5)
        # clip_grad_norm_ 返回未裁剪前的梯度 L2 范数；若为 NaN/Inf（梯度中含 NaN），
        # GradScaler 的 found_inf 标志只检测 Inf 而不检测 NaN，因此需要手动跳过
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
        else:
            print(f'[WARN] epoch {epoch} iter {i}: grad_norm={grad_norm.item()}, skipping optimizer step')
            for p in model_restoration.parameters():
                p.grad = None
        scaler.update()
        epoch_loss += loss.item()
        iter += 1

        # 每100个iteration打印一次
        if i % 100 == 0:
            print('epoch', epoch, 'iter', i)
            print('loss/fft_loss', loss_fft.item())
            print('loss/char_loss', loss_char.item())
            print('loss/edge_loss', loss_edge.item())
            print('loss/kernal_loss', kernal_loss.mean().item())
            print('loss/iter_loss', loss.item())

    if epoch % args.print_epochs == 0:
        print('loss/epoch_loss', epoch_loss, epoch)

    #### Evaluation ####
    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored, _, _ = model_restoration(input_)

            for res, tar in zip(restored[0], target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        print('val/psnr', psnr_val_rgb, epoch)
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))
        with open(log_dir, "a+") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f] \n" % (
                epoch, psnr_val_rgb, best_epoch, best_psnr))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                               epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(log_dir, "a+") as f:
        f.write("------------------------------------------------------------------\n")
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f} \n".format(epoch,
                                                                                       time.time() - epoch_start_time,
                                                                                       epoch_loss,
                                                                                       scheduler.get_lr()[0]))
        f.write("------------------------------------------------------------------\n")

    # 每轮保存权重
    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    # 保存最新的权重（用于恢复训练）
    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))
