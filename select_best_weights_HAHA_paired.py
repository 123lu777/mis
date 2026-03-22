"""
select_best_weights_HAHA_paired.py
===================================
适用于 train_GoPro_HAHA 训练的权重挑选工具（有参考版 / 指定区间版）。

与无参考版（select_best_weights_HAHA.py）的区别：
  · 本脚本需要提供一张清晰参考图和一张对应的模糊图，
    通过计算去模糊结果与参考图之间的 PSNR / SSIM 来评价权重好坏。
  · 支持按 epoch 区间过滤，只评估感兴趣的检查点。
  · 最终输出 Top-10 权重并保存 CSV。

使用方法：
  1. 修改下方 Config 类中的路径和参数；
  2. 在 PyCharm 中右键运行，或在终端执行：
         python select_best_weights_HAHA_paired.py
"""

import os
import re
import cv2
import glob
import shutil
import time
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# =============================================
# 导入带物理先验约束的 HAHA 版本模型
# =============================================
from MISCFilterNet_HAHA import MISCKernelNet_HAHA as myNet
from models.layers_Deform import window_partitionx, window_reversex


# =============================================================================
# ★★★ 在这里修改所有配置参数，右键运行即可 ★★★
# =============================================================================
class Config:
    # --------------------------------------------------
    # 权重目录（train_GoPro_HAHA 保存权重的文件夹）
    # --------------------------------------------------
    checkpoints_dir = r"./checkpoints_haha/GoPro/MISCFilter_HAHA_GoPro"

    # --------------------------------------------------
    # 评估图像对
    #   blur_image_path  —— 模糊输入图像（任意尺寸，推理时会缩放）
    #   ref_image_path   —— 对应的清晰参考图像（与输出对齐时使用）
    # --------------------------------------------------
    blur_image_path = r"./test_images/blur.png"
    ref_image_path  = r"./test_images/sharp.png"

    # --------------------------------------------------
    # 推理分辨率
    #   infer_size  —— 模型推理时的图像尺寸（正方形边长，建议 512）
    #   output_size —— 将模型输出放大到此尺寸后再与参考图对比
    #                  设为 (W, H) 可固定缩放尺寸；设为 None 则自动匹配参考图尺寸
    #   ★ 若参考图为 1024×1024，保持默认 (1024, 1024) 即可。
    #     若参考图为其他分辨率，修改此处或改为 None（自动对齐）。
    # --------------------------------------------------
    infer_size  = 512
    output_size = (1024, 1024)    # 固定放大到 1024×1024 后再比较；改为 None 则自动匹配参考图

    # --------------------------------------------------
    # epoch 评估区间（含端点）
    #   start_epoch / end_epoch —— 只评估此区间内的 model_epoch_*.pth
    #   若需评估全部，令 start_epoch=0, end_epoch=9999999 即可
    # --------------------------------------------------
    start_epoch = 1
    end_epoch   = 9999999

    # --------------------------------------------------
    # 模型配置（需与 train_GoPro_HAHA.py 训练时一致）
    # --------------------------------------------------
    use_deform_in_feat    = True
    use_deform_in_encoder = True
    rigid_smooth_weight   = 0.05
    rotation_aware_weight = 0.03
    flow_gradient_weight  = 0.02

    # --------------------------------------------------
    # 综合得分公式权重（与 MHG 版保持一致）
    #   score = weight_ssim * SSIM + weight_psnr * (PSNR / psnr_norm)
    # --------------------------------------------------
    weight_ssim = 0.7
    weight_psnr = 0.3
    psnr_norm   = 50.0

    # --------------------------------------------------
    # 输出 Top-K 权重（默认 10）
    # output_dir: 将 Top-K 权重复制到该文件夹（None = 不复制）
    # --------------------------------------------------
    top_k      = 10
    output_dir = r"./checkpoints_haha_best_paired"
# =============================================================================


# ---------------------------------------------------------------------------
# 图像 I/O 工具
# ---------------------------------------------------------------------------
def read_image_rgb(path: str) -> np.ndarray:
    """以 RGB 格式读取图像（uint8）。"""
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"无法读取图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_cv2(img: np.ndarray, size: tuple) -> np.ndarray:
    """将图像 resize 到 (width, height)。"""
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# 权重加载（兼容 DataParallel / 单 GPU 保存）
# ---------------------------------------------------------------------------
def load_checkpoint_fixed(model: nn.Module, weights_path: str, device: torch.device):
    """
    加载权重到 model（strict=False，自动过滤形状不匹配的键）。
    同时处理 'module.' 前缀。
    """
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # 去掉 DataParallel 的 'module.' 前缀
    stripped = OrderedDict()
    for k, v in state_dict.items():
        stripped[k.replace('module.', '')] = v

    model_dict = model.state_dict()
    matched = OrderedDict(
        (k, v) for k, v in stripped.items()
        if k in model_dict and v.shape == model_dict[k].shape
    )

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(matched, strict=False)
    else:
        model.load_state_dict(matched, strict=False)


# ---------------------------------------------------------------------------
# 推理器（每次评估重建，避免状态污染）
# ---------------------------------------------------------------------------
class CheckpointDeblurrer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: nn.Module = None

    def setup_model(self, weights_path: str):
        """构建 HAHA 模型并加载指定权重。"""
        cfg = self.cfg
        model = myNet(
            inference=False,
            use_deform_in_feat=cfg.use_deform_in_feat,
            use_deform_in_encoder=cfg.use_deform_in_encoder,
            rigid_smooth_weight=cfg.rigid_smooth_weight,
            rotation_aware_weight=cfg.rotation_aware_weight,
            flow_gradient_weight=cfg.flow_gradient_weight,
        )

        if self.device.type == 'cuda':
            model = model.cuda()
            model = nn.DataParallel(model)
        else:
            model = model.to(self.device)

        load_checkpoint_fixed(model, weights_path, self.device)
        model.eval()
        self.model = model

    def infer(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        对 img_rgb（uint8 RGB）运行 HAHA 推理，返回去模糊结果（uint8 RGB）。
        推理时使用 window_partitionx / window_reversex 以支持任意分辨率。
        """
        inp_np = img_rgb.astype(np.float32) / 255.0
        inp_t  = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        _, _, H, W = inp_t.shape
        win_size = self.cfg.infer_size

        with torch.no_grad():
            input_re, batch_list = window_partitionx(inp_t, win_size)
            # HAHA forward 返回 (outputs, outputs_fil, Kernal_Loss)
            restored, _, _ = self.model(input_re)
            restored = restored[0]
            restored = window_reversex(restored, win_size, H, W, batch_list)
            restored = torch.clamp(restored, 0, 1)

        out = restored.permute(0, 2, 3, 1).cpu().numpy()[0]
        return (out * 255.0).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def compute_metrics(ref_rgb: np.ndarray, out_rgb: np.ndarray):
    """返回 (psnr, ssim)。两张图像必须尺寸相同。"""
    psnr = compare_psnr(ref_rgb, out_rgb)
    ssim = compare_ssim(ref_rgb, out_rgb, channel_axis=2)
    return psnr, ssim


def final_score(psnr: float, ssim: float, cfg: Config) -> float:
    return cfg.weight_ssim * ssim + cfg.weight_psnr * (psnr / cfg.psnr_norm)


# ---------------------------------------------------------------------------
# 主评估逻辑
# ---------------------------------------------------------------------------
def evaluate_all_weights():
    cfg = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 70}")
    print("🧪 开始自动评估 HAHA checkpoint（有参考 / 指定区间版）")
    print(f"{'=' * 70}")
    print(f"使用设备: {device}")

    # ---- 1. 读取参考图和模糊图 ----
    if not os.path.isfile(cfg.ref_image_path):
        print(f"❌ 参考图不存在: {cfg.ref_image_path}")
        return
    if not os.path.isfile(cfg.blur_image_path):
        print(f"❌ 模糊图不存在: {cfg.blur_image_path}")
        return

    ref_image   = read_image_rgb(cfg.ref_image_path)
    blur_image  = read_image_rgb(cfg.blur_image_path)

    ref_h, ref_w = ref_image.shape[:2]
    blur_h, blur_w = blur_image.shape[:2]
    print(f"\n📷 参考图尺寸: {ref_w}×{ref_h}  |  模糊图尺寸: {blur_w}×{blur_h}")
    if cfg.output_size is not None:
        cmp_w, cmp_h = cfg.output_size
        print(f"📐 推理输出将放大到 {cmp_w}×{cmp_h} 后与参考图对比")
        if (ref_h, ref_w) != (cmp_h, cmp_w):
            print(f"   ⚠ 注意：参考图尺寸 {ref_w}×{ref_h} 与对比尺寸 {cmp_w}×{cmp_h} 不一致，"
                  f"请确认 ref_image_path 是否为正确的对应清晰图像！")
    else:
        print(f"📐 推理输出将自动缩放到参考图尺寸 {ref_w}×{ref_h} 后对比")

    # 将模糊图缩放到推理分辨率
    infer_hw = (cfg.infer_size, cfg.infer_size)
    blur_infer = resize_cv2(blur_image, infer_hw)
    print(f"   模糊图已缩放到 {cfg.infer_size}×{cfg.infer_size} 进行推理")

    # ---- 2. 收集并过滤权重文件 ----
    if not os.path.isdir(cfg.checkpoints_dir):
        print(f"❌ 权重目录不存在: {cfg.checkpoints_dir}")
        return

    all_ckpts = glob.glob(os.path.join(cfg.checkpoints_dir, 'model_epoch_*.pth'))
    print(f"\n🔍 文件夹中共发现 {len(all_ckpts)} 个权重文件")

    pattern = re.compile(r'model_epoch_(\d+)\.pth$')
    filtered = []
    for f in all_ckpts:
        m = pattern.search(os.path.basename(f))
        if m:
            epoch = int(m.group(1))
            if cfg.start_epoch <= epoch <= cfg.end_epoch:
                filtered.append((epoch, f))

    if not filtered:
        print(f"\n❌ epoch [{cfg.start_epoch}, {cfg.end_epoch}] 区间内没有找到匹配的权重！")
        return

    # 按 epoch 排序
    filtered.sort(key=lambda x: x[0])
    checkpoint_files = [f for _, f in filtered]

    print(f"🎚 评估区间: epoch {cfg.start_epoch} → {cfg.end_epoch}")
    print(f"🎯 区间内找到 {len(checkpoint_files)} 个权重，开始评估\n")

    # ---- 3. 逐权重推理评估 ----
    all_results = []
    t_start = time.time()

    for i, wpath in enumerate(checkpoint_files, 1):
        wname = os.path.basename(wpath)
        epoch = int(pattern.search(wname).group(1))
        print(f"[{i}/{len(checkpoint_files)}] 测试权重: {wname}", end='', flush=True)
        t0 = time.time()

        try:
            deblurrer = CheckpointDeblurrer(cfg)
            deblurrer.setup_model(wpath)

            restored = deblurrer.infer(blur_infer)

            # 将输出放大到指定尺寸后再与参考图对比
            if cfg.output_size is not None:
                cmp_w, cmp_h = cfg.output_size
                restored_cmp = resize_cv2(restored, cfg.output_size)
                print(f"\n   📐 推理输出已放大到 {cmp_w}×{cmp_h} 以对齐参考图", end='', flush=True)
            elif restored.shape[:2] != ref_image.shape[:2]:
                # 自动匹配参考图尺寸 (H, W) → cv2 需要 (W, H)
                h, w = ref_image.shape[:2]
                restored_cmp = resize_cv2(restored, (w, h))
                print(f"\n   📐 推理输出已自动缩放到 {w}×{h} 以对齐参考图", end='', flush=True)
            else:
                restored_cmp = restored

            psnr, ssim = compute_metrics(ref_image, restored_cmp)
            score = final_score(psnr, ssim, cfg)

            elapsed = time.time() - t0
            avg_sec = (time.time() - t_start) / i
            eta_min = avg_sec * (len(checkpoint_files) - i) / 60.0

            all_results.append({
                'weights': wname,
                'path':    wpath,
                'epoch':   epoch,
                'psnr':    psnr,
                'ssim':    ssim,
                'score':   score,
            })

            print(f"\n   ✔ SSIM={ssim:.4f}, PSNR={psnr:.2f}, Score={score:.4f}"
                  f"  (耗时{elapsed:.1f}s, 预计剩余{eta_min:.1f}分)\n")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n   ⚠ 推理出错，跳过: {e}  (耗时{elapsed:.1f}s)\n")

    if not all_results:
        print("❌ 没有成功评估任何权重！")
        return

    # ---- 4. 排序并输出 Top-K ----
    all_results.sort(key=lambda x: x['score'], reverse=True)
    top_k_results = all_results[:cfg.top_k]

    print('=' * 80)
    print(f"🏆 epoch [{cfg.start_epoch}, {cfg.end_epoch}] 区间内最优 {cfg.top_k} 个权重：")
    print('=' * 80)
    for rank, r in enumerate(top_k_results, 1):
        flag = " ★" if rank == 1 else ""
        print(f"  [{rank:>2}] {r['weights']:<40} epoch={r['epoch']:<6} "
              f"Score={r['score']:.4f}  SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f}{flag}")

    print(f"\n🎯 【最终结果：最优 {cfg.top_k} 个权重名称】")
    print('-' * 60)
    for r in top_k_results:
        print(r['weights'])

    # ---- 5. 复制 Top-K 权重 ----
    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"\n{'=' * 65}")
        print(f"复制 Top-{cfg.top_k} 权重到: {cfg.output_dir}")
        for rank, r in enumerate(top_k_results, 1):
            dst = os.path.join(cfg.output_dir, f"rank{rank:02d}_{r['weights']}")
            shutil.copy2(r['path'], dst)
            print(f"  [{rank}] {r['weights']}  Score={r['score']:.4f}")

    # ---- 6. 保存 CSV ----
    csv_path = os.path.join(cfg.checkpoints_dir, 'weight_selection_paired_results.csv')
    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('rank,weights_file,epoch,score,ssim,psnr\n')
            for rank, r in enumerate(all_results, 1):
                f.write(f"{rank},{r['weights']},{r['epoch']},"
                        f"{r['score']:.6f},{r['ssim']:.4f},{r['psnr']:.2f}\n")
        print(f"\n📊 评估结果已保存到: {csv_path}")
    except Exception as e:
        print(f"⚠ CSV 写入失败: {e}")

    print("\nDone.\n")
    return top_k_results


# =============================================================================
# PyCharm 直接运行入口
# =============================================================================
def main():
    evaluate_all_weights()
    input("按回车键退出...")


if __name__ == '__main__':
    main()
