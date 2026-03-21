import os
import sys
import cv2
import glob
import math
import shutil
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# =============================================
# 导入带物理先验约束的 HAHA 版本模型
# =============================================
from MISCFilterNet_HAHA import MISCKernelNet_HAHA as myNet
from models.layers_Deform import window_partitionx, window_reversex

try:
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


# =============================================================================
# ★★★ 在这里修改所有配置参数，右键运行即可 ★★★
# =============================================================================
class Config:
    # --------------------------------------------------
    # 权重目录（train_GoPro_HAHA 保存权重的文件夹）
    # --------------------------------------------------
    checkpoints_dir = r"./checkpoints_haha/GoPro/MISCFilter_HAHA_GoPro"

    # --------------------------------------------------
    # 验证图像目录
    # 结构要求（与 GoPro Large 数据集一致）：
    #   val_input_dir/  —— 模糊图像（.png / .jpg）
    #   val_target_dir/ —— 清晰图像（.png / .jpg，文件名与输入一一对应）
    # 若你没有 sharp/blur 对，也可以把 val_target_dir 设成与 val_input_dir
    # 相同，此时 PSNR 固定为 inf，程序仅作为推理速度测试。
    # --------------------------------------------------
    val_input_dir  = r"./dataset/GOPRO_Large/test/blur"
    val_target_dir = r"./dataset/GOPRO_Large/test/sharp"

    # --------------------------------------------------
    # 评估用最多图像数（设为 None 则使用全部验证图像）
    # 减小该数值可加快评估速度，但排名精度降低
    # --------------------------------------------------
    max_eval_images = 50

    # --------------------------------------------------
    # 模型配置（需与训练时 Config 中一致）
    # --------------------------------------------------
    use_deform_in_feat    = True
    use_deform_in_encoder = True
    rigid_smooth_weight   = 0.05
    rotation_aware_weight = 0.03
    flow_gradient_weight  = 0.02

    # --------------------------------------------------
    # 推理窗口大小（与测试脚本保持一致）
    # --------------------------------------------------
    win_size = 512

    # --------------------------------------------------
    # 结果输出
    # top_k: 保留 PSNR 最高的前 K 个权重
    # output_dir: 将 top_k 权重复制到该文件夹（设为 None 则不复制）
    # --------------------------------------------------
    top_k      = 5
    output_dir = r"./checkpoints_haha_best"

    # --------------------------------------------------
    # 是否同时评估 SSIM（需要 scikit-image，速度稍慢）
    # --------------------------------------------------
    eval_ssim = True

    # --------------------------------------------------
    # 是否包含 model_best.pth 和 model_latest.pth（如果存在）
    # --------------------------------------------------
    include_special = True
# =============================================================================


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def psnr_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张 uint8 图像的 PSNR（单位 dB）"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张 uint8 RGB 图像的平均 SSIM"""
    if not HAS_SSIM:
        return float('nan')
    val = ssim_func(img1, img2, channel_axis=2, data_range=255)
    return float(val)


def load_image_pairs(input_dir: str, target_dir: str, max_n=None):
    """
    返回 [(input_path, target_path), ...] 列表。
    文件名需在两个目录下一一对应（允许不同扩展名）。
    """
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    input_files = []
    for ext in exts:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))
    input_files = sorted(set(input_files))

    if not input_files:
        # 尝试递归子目录（GoPro 数据集按视频分文件夹）
        for ext in exts:
            input_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        input_files = sorted(set(input_files))

    pairs = []
    for inp in input_files:
        rel = os.path.relpath(inp, input_dir)
        # 优先同名同扩展名；若不存在，尝试常见扩展名
        tgt = os.path.join(target_dir, rel)
        if not os.path.exists(tgt):
            base = os.path.splitext(rel)[0]
            found = False
            for ext in ('.png', '.jpg', '.jpeg'):
                candidate = os.path.join(target_dir, base + ext)
                if os.path.exists(candidate):
                    tgt = candidate
                    found = True
                    break
            if not found:
                continue
        pairs.append((inp, tgt))

    if max_n is not None:
        pairs = pairs[:max_n]
    return pairs


# ---------------------------------------------------------------------------
# 模型构建与权重加载
# ---------------------------------------------------------------------------
def build_model(cfg: Config, device: torch.device) -> nn.Module:
    """构建 HAHA 模型（不加载权重）"""
    model = myNet(
        inference=False,
        use_deform_in_feat=cfg.use_deform_in_feat,
        use_deform_in_encoder=cfg.use_deform_in_encoder,
        rigid_smooth_weight=cfg.rigid_smooth_weight,
        rotation_aware_weight=cfg.rotation_aware_weight,
        flow_gradient_weight=cfg.flow_gradient_weight,
    )
    if device.type == 'cuda':
        model = model.cuda()
        model = nn.DataParallel(model)
    else:
        model = model.to(device)
    model.eval()
    return model


def load_weights(model: nn.Module, weights_path: str, device: torch.device):
    """加载权重文件到模型（兼容 DataParallel 和单 GPU）"""
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # 去掉 DataParallel 前缀
    new_state = {}
    for k, v in state_dict.items():
        new_state[k.replace('module.', '')] = v

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(new_state, strict=False)
    else:
        model.load_state_dict(new_state, strict=False)


# ---------------------------------------------------------------------------
# 单张图像推理
# ---------------------------------------------------------------------------
def infer_single(model: nn.Module, image_path: str, win_size: int, device: torch.device) -> np.ndarray:
    """
    对单张模糊图像运行 HAHA 推理，返回去模糊结果（uint8 RGB numpy）。
    """
    inp = Image.open(image_path).convert('RGB')
    inp_np = np.array(inp).astype(np.float32) / 255.0
    inp_t = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(device)

    _, _, H, W = inp_t.shape
    with torch.no_grad():
        input_re, batch_list = window_partitionx(inp_t, win_size)
        # HAHA forward 返回三元组：(outputs, outputs_fil, Kernal_Loss)
        restored, _, _ = model(input_re)
        restored = restored[0]
        restored = window_reversex(restored, win_size, H, W, batch_list)
        restored = torch.clamp(restored, 0, 1)

    out = restored.permute(0, 2, 3, 1).cpu().numpy()[0]
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return out  # RGB


# ---------------------------------------------------------------------------
# 单个权重文件评估
# ---------------------------------------------------------------------------
def evaluate_checkpoint(model: nn.Module, weights_path: str,
                         pairs: list, cfg: Config,
                         device: torch.device) -> dict:
    """
    加载 weights_path 到 model，在 pairs 上计算 PSNR（和 SSIM），
    返回 {'weights': ..., 'psnr': float, 'ssim': float, 'n_images': int}。
    """
    try:
        load_weights(model, weights_path, device)
    except Exception as e:
        print(f"  ⚠ 加载权重失败 [{os.path.basename(weights_path)}]: {e}")
        return {'weights': weights_path, 'psnr': -1.0, 'ssim': -1.0, 'n_images': 0}

    psnr_list = []
    ssim_list = []

    for inp_path, tgt_path in pairs:
        try:
            pred_rgb = infer_single(model, inp_path, cfg.win_size, device)
            tgt_bgr = cv2.imread(tgt_path)
            if tgt_bgr is None:
                continue
            tgt_rgb = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB)

            # 如果尺寸不一致，缩放预测结果以匹配目标尺寸
            if pred_rgb.shape[:2] != tgt_rgb.shape[:2]:
                pred_rgb = cv2.resize(pred_rgb, (tgt_rgb.shape[1], tgt_rgb.shape[0]),
                                      interpolation=cv2.INTER_LANCZOS4)

            psnr_list.append(psnr_numpy(pred_rgb, tgt_rgb))
            if cfg.eval_ssim and HAS_SSIM:
                ssim_list.append(ssim_numpy(pred_rgb, tgt_rgb))

        except Exception as e:
            print(f"    ⚠ 推理出错 [{os.path.basename(inp_path)}]: {e}")
            continue

    n = len(psnr_list)
    avg_psnr = float(np.mean(psnr_list)) if n > 0 else -1.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else float('nan')

    return {'weights': weights_path, 'psnr': avg_psnr, 'ssim': avg_ssim, 'n_images': n}


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    cfg = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ---- 1. 收集权重文件 ----
    ckpt_dir = cfg.checkpoints_dir
    if not os.path.isdir(ckpt_dir):
        print(f"❌ 权重目录不存在: {ckpt_dir}")
        sys.exit(1)

    weight_files = sorted(glob.glob(os.path.join(ckpt_dir, 'model_epoch_*.pth')))

    if cfg.include_special:
        for special in ('model_best.pth', 'model_latest.pth'):
            p = os.path.join(ckpt_dir, special)
            if os.path.exists(p):
                weight_files.append(p)

    if not weight_files:
        print(f"❌ 在 {ckpt_dir} 中没有找到任何权重文件！")
        sys.exit(1)

    print(f"找到 {len(weight_files)} 个权重文件")

    # ---- 2. 收集验证图像对 ----
    if not os.path.isdir(cfg.val_input_dir):
        print(f"❌ 验证输入目录不存在: {cfg.val_input_dir}")
        sys.exit(1)
    if not os.path.isdir(cfg.val_target_dir):
        print(f"❌ 验证目标目录不存在: {cfg.val_target_dir}")
        sys.exit(1)

    pairs = load_image_pairs(cfg.val_input_dir, cfg.val_target_dir, max_n=cfg.max_eval_images)
    if not pairs:
        print("❌ 未找到有效的（输入, 目标）图像对，请检查路径！")
        sys.exit(1)

    print(f"评估图像对数: {len(pairs)}")
    print(f"{'=' * 60}")

    # ---- 3. 构建模型（只构建一次，复用） ----
    model = build_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"{'=' * 60}")

    # ---- 4. 逐权重评估 ----
    results = []
    for idx, wf in enumerate(weight_files, 1):
        name = os.path.basename(wf)
        print(f"[{idx:>3}/{len(weight_files)}] 评估: {name}")
        r = evaluate_checkpoint(model, wf, pairs, cfg, device)
        results.append(r)
        ssim_str = f"  SSIM={r['ssim']:.4f}" if not math.isnan(r['ssim']) else ""
        print(f"        PSNR={r['psnr']:.4f} dB{ssim_str}  (图像数={r['n_images']})")

    # ---- 5. 排序并打印结果 ----
    results.sort(key=lambda x: x['psnr'], reverse=True)

    print(f"\n{'=' * 60}")
    print(f"{'排名':<6} {'权重文件名':<40} {'PSNR (dB)':<14} {'SSIM':<10} {'图像数'}")
    print(f"{'-' * 6} {'-' * 40} {'-' * 14} {'-' * 10} {'-' * 6}")
    for rank, r in enumerate(results, 1):
        name = os.path.basename(r['weights'])
        ssim_str = f"{r['ssim']:.4f}" if not math.isnan(r['ssim']) else "  N/A"
        flag = " ★" if rank <= cfg.top_k else ""
        print(f"{rank:<6} {name:<40} {r['psnr']:<14.4f} {ssim_str:<10} {r['n_images']}{flag}")

    # ---- 6. 保存 top-K 权重 ----
    top_results = results[:cfg.top_k]

    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"复制 Top-{cfg.top_k} 权重到: {cfg.output_dir}")
        for rank, r in enumerate(top_results, 1):
            src = r['weights']
            dst = os.path.join(cfg.output_dir, f"rank{rank:02d}_{os.path.basename(src)}")
            shutil.copy2(src, dst)
            print(f"  [{rank}] {os.path.basename(src)} → {os.path.basename(dst)}")

    # ---- 7. 最终推荐 ----
    best = results[0]
    print(f"\n{'=' * 60}")
    print(f"🏆 最优权重: {os.path.basename(best['weights'])}")
    print(f"   PSNR = {best['psnr']:.4f} dB")
    if not math.isnan(best['ssim']):
        print(f"   SSIM = {best['ssim']:.4f}")
    print(f"   路径: {best['weights']}")
    print(f"{'=' * 60}")

    # 将结果写入 CSV 方便后续分析
    csv_path = os.path.join(ckpt_dir, 'weight_selection_results.csv')
    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('rank,weights_file,psnr_db,ssim,n_images\n')
            for rank, r in enumerate(results, 1):
                ssim_val = f"{r['ssim']:.6f}" if not math.isnan(r['ssim']) else ''
                f.write(f"{rank},{os.path.basename(r['weights'])},{r['psnr']:.6f},{ssim_val},{r['n_images']}\n")
        print(f"📊 评估结果已保存到: {csv_path}")
    except Exception as e:
        print(f"⚠ CSV 写入失败: {e}")


if __name__ == '__main__':
    main()
