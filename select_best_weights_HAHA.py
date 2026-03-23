import os
import sys
import cv2
import glob
import math
import shutil
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

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
    # ★ 只需输入模糊图像文件夹，无需参考清晰图像 ★
    # 程序对每个权重推理这些模糊图像，用无参考清晰度指标衡量去模糊效果。
    # 支持单层目录或递归子目录（GoPro 按视频分文件夹均可）。
    # --------------------------------------------------
    blur_input_dir = r"./dataset/GOPRO_Large/test/blur"

    # --------------------------------------------------
    # 每个权重最多评估多少张图（None = 全部）
    # 减小该数值可加快速度，建议先用 20~30 张快速筛选
    # --------------------------------------------------
    max_eval_images = 30

    # --------------------------------------------------
    # 模型配置（需与 train_GoPro_HAHA.py 训练时一致）
    # --------------------------------------------------
    use_deform_in_feat    = True
    use_deform_in_encoder = True
    rigid_smooth_weight   = 0.05
    rotation_aware_weight = 0.03
    flow_gradient_weight  = 0.02
    curl_smooth_weight    = 0.04

    # --------------------------------------------------
    # 推理窗口大小
    # --------------------------------------------------
    win_size = 512

    # --------------------------------------------------
    # 综合得分各项无参考指标的权重（三项之和建议为 1.0）
    #
    #   laplacian  —— 拉普拉斯方差：反映高频细节清晰度（越高越好）
    #   tenengrad  —— Tenengrad 梯度能量：反映边缘锐利程度（越高越好）
    #   brenner    —— Brenner 梯度：对焦距离变化敏感（越高越好）
    #
    # 三项指标在所有权重上做 Min-Max 归一化后加权求和，
    # 综合得分范围为 [0, 1]，越高代表去模糊效果越好。
    # --------------------------------------------------
    weight_laplacian = 0.4
    weight_tenengrad = 0.4
    weight_brenner   = 0.2

    # --------------------------------------------------
    # 输出 Top-K 权重（默认 10）
    # output_dir: 将 Top-K 权重复制到该文件夹（None = 不复制）
    # --------------------------------------------------
    top_k      = 10
    output_dir = r"./checkpoints_haha_best"

    # --------------------------------------------------
    # 是否包含 model_best.pth 和 model_latest.pth（若存在）
    # --------------------------------------------------
    include_special = True

    # --------------------------------------------------
    # 常量输出检测阈值
    # 若某权重的三项指标均值都低于此阈值，视为该权重输出了
    # 常量图像（全黑/全白）——通常是训练极早期尚未收敛的权重。
    # 这类权重会被标记为 [常量输出]，从归一化基准和 Top-K 中排除。
    # 若需关闭此检测，可将值设为 0.0。
    # --------------------------------------------------
    constant_output_threshold = 0.5
# =============================================================================


# ---------------------------------------------------------------------------
# 无参考清晰度指标
# ---------------------------------------------------------------------------
def laplacian_score(gray: np.ndarray) -> float:
    """
    拉普拉斯方差：对全图二阶导数求方差。
    越清晰的图像高频成分越多，方差越大。
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad_score(gray: np.ndarray) -> float:
    """
    Tenengrad 梯度能量：Sobel 梯度幅值平方的均值。
    边缘越锐利，值越大。
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def brenner_score(gray: np.ndarray) -> float:
    """
    Brenner 梯度：相邻两像素差值平方的均值（水平方向）。
    对模糊非常敏感，是经典焦距评估指标之一。
    """
    diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
    return float(np.mean(diff ** 2))


def sharpness_metrics(img_rgb: np.ndarray) -> dict:
    """
    给定 RGB 图像（uint8 或 float32 [0,1]），计算三项清晰度指标。
    返回 {'laplacian': float, 'tenengrad': float, 'brenner': float}
    """
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255.0).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return {
        'laplacian': laplacian_score(gray),
        'tenengrad': tenengrad_score(gray),
        'brenner':   brenner_score(gray),
    }


# ---------------------------------------------------------------------------
# 图像收集
# ---------------------------------------------------------------------------
def collect_images(input_dir: str, max_n=None) -> list:
    """
    收集 input_dir（及其子目录）中的所有图像路径，最多返回 max_n 张。
    """
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files = sorted(set(files))

    if not files:
        # 递归搜索子目录
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        files = sorted(set(files))

    if max_n is not None:
        files = files[:max_n]
    return files


# ---------------------------------------------------------------------------
# 模型构建与权重加载
# ---------------------------------------------------------------------------
def build_model(cfg: Config, device: torch.device) -> nn.Module:
    """构建 HAHA 模型（不加载权重，只构建一次后复用）"""
    model = myNet(
        inference=True,
        use_deform_in_feat=cfg.use_deform_in_feat,
        use_deform_in_encoder=cfg.use_deform_in_encoder,
        rigid_smooth_weight=cfg.rigid_smooth_weight,
        rotation_aware_weight=cfg.rotation_aware_weight,
        flow_gradient_weight=cfg.flow_gradient_weight,
        curl_smooth_weight=cfg.curl_smooth_weight,
    )
    if device.type == 'cuda':
        model = model.cuda()
        model = nn.DataParallel(model)
    else:
        model = model.to(device)
    model.eval()
    return model


def load_weights(model: nn.Module, weights_path: str, device: torch.device):
    """加载权重到模型（兼容单 GPU / DataParallel / 多 GPU 保存的权重）"""
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

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
def infer_single(model: nn.Module, image_path: str,
                 win_size: int, device: torch.device) -> np.ndarray:
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
        restored = model(input_re)
        restored = window_reversex(restored, win_size, H, W, batch_list)
        restored = torch.clamp(restored, 0, 1)

    out = restored.permute(0, 2, 3, 1).cpu().numpy()[0]
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return out  # RGB


# ---------------------------------------------------------------------------
# 单个权重文件评估（无参考模式）
# ---------------------------------------------------------------------------
def evaluate_checkpoint(model: nn.Module, weights_path: str,
                         image_files: list, cfg: Config,
                         device: torch.device) -> dict:
    """
    加载 weights_path，对 image_files 中每张图像推理后计算无参考清晰度指标，
    返回各项指标在所有图像上的均值。
    """
    try:
        load_weights(model, weights_path, device)
    except Exception as e:
        print(f"  ⚠ 加载权重失败 [{os.path.basename(weights_path)}]: {e}")
        return {
            'weights': weights_path,
            'laplacian': 0.0, 'tenengrad': 0.0, 'brenner': 0.0,
            'composite': 0.0, 'n_images': 0, 'is_constant': True,
        }

    lap_list, ten_list, bre_list = [], [], []

    for img_path in image_files:
        try:
            pred_rgb = infer_single(model, img_path, cfg.win_size, device)
            m = sharpness_metrics(pred_rgb)
            lap_list.append(m['laplacian'])
            ten_list.append(m['tenengrad'])
            bre_list.append(m['brenner'])
        except Exception as e:
            print(f"    ⚠ 推理出错 [{os.path.basename(img_path)}]: {e}")
            continue

    n = len(lap_list)
    avg_lap = float(np.mean(lap_list)) if n > 0 else 0.0
    avg_ten = float(np.mean(ten_list)) if n > 0 else 0.0
    avg_bre = float(np.mean(bre_list)) if n > 0 else 0.0

    # 常量输出检测：三项指标均低于阈值说明模型输出了全零/全常数图像
    thr = cfg.constant_output_threshold
    is_constant = (avg_lap <= thr and avg_ten <= thr and avg_bre <= thr)

    return {
        'weights':     weights_path,
        'laplacian':   avg_lap,
        'tenengrad':   avg_ten,
        'brenner':     avg_bre,
        'composite':   0.0,   # 将在所有权重评估完后统一计算
        'n_images':    n,
        'is_constant': is_constant,
    }


# ---------------------------------------------------------------------------
# 计算综合得分（跨所有权重归一化后加权）
# ---------------------------------------------------------------------------
def compute_composite_scores(results: list, cfg: Config):
    """
    对 results 中三项指标做 Min-Max 归一化（[0,1]），
    按配置权重加权后写入 composite 字段。

    归一化基准仅使用"有效"权重（非常量输出）。
    常量输出权重的 composite 固定为 0.0，排名自然垫底。
    """
    valid = [r for r in results if not r.get('is_constant', False)]
    if not valid:
        # 全部都是常量输出（极端情况），退化为全量归一化
        valid = results

    def _normalize_with_ref(vals_all, ref_vals, name):
        """用 ref_vals 计算 min/max，对 vals_all 做归一化。"""
        mn, mx = min(ref_vals), max(ref_vals)
        if mx - mn < 1e-12:
            print(f"  ⚠ 指标 [{name}] 在所有有效权重上几乎相同（range={mx-mn:.2e}），归一化后统一为 0.5")
            # 有效权重统一给 0.5，常量输出给 0.0
            return [0.5 if not r.get('is_constant', False) else 0.0
                    for r in results]
        return [
            0.0 if r.get('is_constant', False)
            else max(0.0, (v - mn) / (mx - mn))
            for r, v in zip(results, vals_all)
        ]

    lap_vals = [r['laplacian'] for r in results]
    ten_vals = [r['tenengrad'] for r in results]
    bre_vals = [r['brenner']   for r in results]

    lap_ref = [r['laplacian'] for r in valid]
    ten_ref = [r['tenengrad'] for r in valid]
    bre_ref = [r['brenner']   for r in valid]

    lap_norm = _normalize_with_ref(lap_vals, lap_ref, 'laplacian')
    ten_norm = _normalize_with_ref(ten_vals, ten_ref, 'tenengrad')
    bre_norm = _normalize_with_ref(bre_vals, bre_ref, 'brenner')

    wl, wt, wb = cfg.weight_laplacian, cfg.weight_tenengrad, cfg.weight_brenner
    for i, r in enumerate(results):
        if r.get('is_constant', False):
            r['composite'] = 0.0
        else:
            r['composite'] = wl * lap_norm[i] + wt * ten_norm[i] + wb * bre_norm[i]


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

    # ---- 2. 收集模糊图像 ----
    if not os.path.isdir(cfg.blur_input_dir):
        print(f"❌ 模糊图像目录不存在: {cfg.blur_input_dir}")
        sys.exit(1)

    image_files = collect_images(cfg.blur_input_dir, max_n=cfg.max_eval_images)
    if not image_files:
        print(f"❌ 在 {cfg.blur_input_dir} 中未找到任何图像！")
        sys.exit(1)

    print(f"评估图像数: {len(image_files)}")
    print(f"综合得分权重 — 拉普拉斯:{cfg.weight_laplacian}  "
          f"Tenengrad:{cfg.weight_tenengrad}  Brenner:{cfg.weight_brenner}")
    print(f"{'=' * 65}")

    # ---- 3. 构建模型（只构建一次，复用） ----
    model = build_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"{'=' * 65}")

    # ---- 4. 逐权重评估 ----
    results = []
    t_start_all = time.time()
    for idx, wf in enumerate(weight_files, 1):
        name = os.path.basename(wf)
        print(f"[{idx:>3}/{len(weight_files)}] 评估: {name}", end='', flush=True)
        t0 = time.time()
        r = evaluate_checkpoint(model, wf, image_files, cfg, device)
        elapsed = time.time() - t0
        results.append(r)

        # 每次评估后重新计算 ETA
        avg_sec = (time.time() - t_start_all) / idx
        remaining = len(weight_files) - idx
        eta_min = avg_sec * remaining / 60.0

        if r.get('is_constant', False):
            print(f"\n        ⚠ [常量输出] 该权重推理结果为全零/常数图像，将排除于归一化和 Top-K"
                  f"  (耗时{elapsed:.1f}s, 预计剩余{eta_min:.1f}分)")
        else:
            print(f"\n        Laplacian={r['laplacian']:.2f}  "
                  f"Tenengrad={r['tenengrad']:.2f}  "
                  f"Brenner={r['brenner']:.2f}  "
                  f"(图像数={r['n_images']}, 耗时{elapsed:.1f}s, 预计剩余{eta_min:.1f}分)")

    # ---- 5. 计算综合得分并排序 ----
    compute_composite_scores(results, cfg)
    results.sort(key=lambda x: x['composite'], reverse=True)

    n_constant = sum(1 for r in results if r.get('is_constant', False))
    n_valid = len(results) - n_constant
    print(f"\n{'=' * 65}")
    print(f"有效权重: {n_valid} 个  |  常量输出（已排除归一化）: {n_constant} 个")

    # ---- 6. 打印结果表格 ----
    print(f"{'=' * 65}")
    hdr = f"{'排名':<5} {'权重文件名':<38} {'综合':<8} {'拉普拉斯':<12} {'Tenengrad':<12} {'Brenner':<10} {'备注'}"
    print(hdr)
    print('-' * len(hdr))
    rank_valid = 0
    for r in results:
        name = os.path.basename(r['weights'])
        if r.get('is_constant', False):
            print(
                f"{'--':<5} {name:<38} {'--':<8} "
                f"{'--':<12} {'--':<12} {'--':<10} [常量输出]"
            )
        else:
            rank_valid += 1
            flag = " ★" if rank_valid <= cfg.top_k else ""
            print(
                f"{rank_valid:<5} {name:<38} {r['composite']:.4f}  "
                f"{r['laplacian']:<12.2f} {r['tenengrad']:<12.2f} "
                f"{r['brenner']:<10.2f}{flag}"
            )

    # ---- 7. 复制 Top-K 权重（跳过常量输出） ----
    valid_results = [r for r in results if not r.get('is_constant', False)]
    top_results = valid_results[:cfg.top_k]

    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"\n{'=' * 65}")
        print(f"复制 Top-{cfg.top_k} 权重到: {cfg.output_dir}")
        for rank, r in enumerate(top_results, 1):
            src = r['weights']
            dst = os.path.join(cfg.output_dir, f"rank{rank:02d}_{os.path.basename(src)}")
            shutil.copy2(src, dst)
            print(f"  [{rank}] {os.path.basename(src)}  综合={r['composite']:.4f}")

    # ---- 8. 最终推荐 ----
    best = valid_results[0] if valid_results else results[0]
    print(f"\n{'=' * 65}")
    print(f"🏆 最优权重: {os.path.basename(best['weights'])}")
    print(f"   综合得分 = {best['composite']:.4f}")
    print(f"   Laplacian  = {best['laplacian']:.2f}")
    print(f"   Tenengrad  = {best['tenengrad']:.2f}")
    print(f"   Brenner    = {best['brenner']:.2f}")
    print(f"   路径: {best['weights']}")
    print(f"{'=' * 65}")

    # ---- 9. 保存 CSV ----
    csv_path = os.path.join(ckpt_dir, 'weight_selection_results.csv')
    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('rank,weights_file,composite_score,laplacian,tenengrad,brenner,n_images,is_constant\n')
            rank_valid = 0
            for r in results:
                if r.get('is_constant', False):
                    rank_str = ''
                    comp_str = ''
                else:
                    rank_valid += 1
                    rank_str = str(rank_valid)
                    comp_str = f"{r['composite']:.6f}"
                f.write(
                    f"{rank_str},{os.path.basename(r['weights'])},"
                    f"{comp_str},{r['laplacian']:.4f},"
                    f"{r['tenengrad']:.4f},{r['brenner']:.4f},"
                    f"{r['n_images']},{int(r.get('is_constant', False))}\n"
                )
        print(f"📊 评估结果已保存到: {csv_path}")
    except Exception as e:
        print(f"⚠ CSV 写入失败: {e}")


if __name__ == '__main__':
    main()
