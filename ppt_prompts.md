# PPT 生成提示词：MISCFilter 原论文 vs 本仓库改进版对比

> 将以下提示词（或其中某段）复制给 AI（如 ChatGPT / Kimi / 文心一言 / Gamma / Beautiful.ai）即可生成 PPT 大纲或完整幻灯片。

---

## 【提示词 1】完整中文 PPT 大纲（适合粘贴给任意 AI）

```
请为我生成一份学术汇报 PPT 的详细大纲，主题是：
「MISCFilter 图像去模糊网络的改进研究：可变形卷积 + 光流物理先验约束」

PPT 共 12～15 页，内容按以下结构组织，每页给出标题、3～5 条要点和推荐的配图/示意图建议：

第 1 页  封面
- 标题：基于光流物理先验约束的 MISCFilter 图像去模糊改进研究
- 副标题：MISCFilterNet-HAHA：可变形卷积 + 三项物理约束
- 作者/日期

第 2 页  研究背景与动机
- 图像去模糊的应用场景（运动模糊、风机叶片旋转模糊）
- 现有方法的不足：对刚性旋转模糊估计不准确
- 本工作的出发点：加入物理先验让光流更可靠

第 3 页  原始 MISCFilter 论文核心思想
- 全称：Multi-Input Single-Content Filter Network
- 核心组件：SCM（浅层特征提取）、FAM（特征注意力融合）、AFF（跨尺度特征融合）
- 三尺度多分辨率架构（1×、1/2×、1/4×）
- 基于光流的核预测（KernelPredict Flow + Warp 融合）
- 损失函数：Charbonnier Loss + 频域 FFT Loss

第 4 页  第一步改进：可变形卷积（MISCFilterNet-Deform）
- 将所有标准卷积替换为 Deformable Convolution（DCNv2）
- 改进目标：自适应感受野，更好捕获旋转/弯曲模糊
- 改进模块：EBlock_Deform、DBlock_Deform、SCM_Deform、FAM_Deform、AFF_Deform
- ResBlock_Deform_fft_bench：残差块内集成 FFT 频域特征
- 代价：显存占用提升，建议 batch_size 从 16 降至 8

第 5 页  第二步改进：光流物理先验约束（MISCFilterNet-HAHA）
- 核心思想：在光流预测阶段加入三项物理约束，无需旋转中心标注、无需分割掩码
- 约束一 — 刚性运动平滑：L_smooth = Σ√(|∇u|²+ε²) + √(|∇v|²+ε²)
  → 刚性物体相邻像素光流差异应趋近于零
- 约束二 — 旋转感知：L_rotation = mean(clamp(div − curl, min=0))
  → 旋转运动特征：curl（旋度）大、div（散度）小；惩罚散开大于旋转的情况
- 约束三 — 二阶梯度平滑：L_gradient = Σ√(|∇²u|²+ε²) + √(|∇²v|²+ε²)
  → 不仅光流本身平滑，光流的变化率也应平滑，防止抖动/碎裂
- 三项约束在三个尺度（s1/s2/s3）分别累加进 Kernal_Loss

第 6 页  物理约束数学细节
- 散度（divergence）：div = ∂u/∂x + ∂v/∂y —— 度量流场膨胀/收缩程度
- 旋度（curl）：curl = ∂v/∂x − ∂u/∂y —— 度量流场旋转程度
- 旋转运动满足：curl >> div；平移满足：curl ≈ 0；纯膨胀满足：div >> curl
- 损失设计亮点：使用 clamp(div−curl, min=0) 仅惩罚"散开 > 旋转"的情况
- 全部偏导数采用前向差分离散近似，完全可微

第 7 页  总损失函数构成
- 原有损失：L_char（Charbonnier）+ L_fft（频域）+ L_edge（边缘）
- 新增物理约束损失（累加到 Kernal_Loss）：
    Kernal_Loss += 0.05 × L_smooth + 0.03 × L_rotation + 0.02 × L_gradient
- 三个尺度 × 三项约束 = 9 个附加约束项
- 总损失 = L_char + λ_fft × L_fft + λ_edge × L_edge + Kernal_Loss

第 8 页  训练工程优化
- AMP 混合精度（FP16）：torch.amp.autocast + GradScaler，4090 上约提速 1.5～2×
- TF32 加速：torch.backends.cudnn.allow_tf32 = True，Ampere 架构免费加速
- Batch size 提升：8 → 12（双卡，每卡 6 张）
- 梯度裁剪：clip_grad_norm_(max_norm=0.5)，防止 FP16 下梯度爆炸
- NaN/Inf 守卫：检测非有限梯度范数，跳过损坏 batch 保护优化器状态
- 数据增强：随机裁剪、翻转、颜色抖动（亮度/饱和度/Gamma）

第 9 页  评估流程修复（Bug Fix）
- 问题描述：所有 epoch 权重均得 SSIM=0.0003、PSNR=5.01（完全相同）
- 根因：DataParallel 包装后 model.state_dict() 的键含 'module.' 前缀，
  而加载时已去除前缀的 stripped 键与之不匹配 → matched 字典为空 →
  每次加载均使用随机初始化权重
- 修复方案：改用 base_model = model.module if DataParallel else model，
  取底层模型的 state_dict() 进行键比对，确保前缀一致
- 附加改进：推理改为 inference=True 模式（forward 直接返回张量，跳过多尺度列表解包和 Kernal_Loss 计算）

第 10 页  网络架构对比表
（建议以表格形式展示）
| 维度 | MISCFilter（原论文）| Deform 版 | HAHA 版（本工作）|
|---|---|---|---|
| 卷积类型 | 标准卷积 | 可变形卷积 | 可变形卷积 |
| 物理约束 | 无 | 无 | 3 项光流约束 |
| 损失组成 | Char + FFT + Edge | 同左 | 同左 + Kernal_Loss |
| 训练精度 | FP32 | FP32 | AMP FP16 |
| 旋转感知 | 无 | 弱 | 强（curl>div 约束）|
| 需要中心标注 | — | — | 否 |

第 11 页  实验设置
- 数据集：GoPro Large（2103 训练对 / 1111 测试对）
- 评估指标：PSNR（dB）、SSIM、综合得分 Score = 0.7×SSIM + 0.3×(PSNR/50)
- 训练轮次：6000 epochs，Cosine Annealing + Warmup（3 epochs）
- 硬件：双 NVIDIA GPU（推荐 RTX 4090 × 2）

第 12 页  结论与展望
- 贡献 1：将可变形卷积引入 MISCFilter，提升对非均匀模糊的感受野适应性
- 贡献 2：设计三项无监督光流物理约束，专为刚性旋转模糊设计，即插即用
- 贡献 3：集成 AMP 混合精度训练，显著提升训练效率
- 修复权重评估流水线的 DataParallel 键匹配 Bug，确保公正评估
- 未来展望：扩展到视频去模糊、验证在真实风机叶片数据上的效果

第 13 页  参考文献
- 原始 MISCFilter 论文（请补充具体引用）
- Deformable Convolutional Networks（Dai et al., ICCV 2017）
- Horn-Schunck Optical Flow（物理约束思想来源）
- PyTorch AMP 官方文档

请为每页生成清晰的要点句，语言为中文学术风格，适合课题组内部汇报。
```

---

## 【提示词 2】英文 PPT（适合 Gamma / Tome / Beautiful.ai）

```
Create a 13-slide academic presentation PPT titled:
"Improving MISCFilter Image Deblurring with Physical Prior Constraints on Optical Flow"

Subtitle: MISCFilterNet-HAHA: Deformable Convolutions + Three Physical Constraints

Slide structure:
1. Title slide
2. Motivation — rotational blur in rigid scenes (e.g., wind turbine blades)
3. Original MISCFilter: multi-scale architecture, SCM/FAM/AFF modules, kernel prediction via flow warping
4. Extension 1 — Deformable Convolutions: adaptive receptive field, DCNv2 in all blocks
5. Extension 2 — Physical Prior Constraints: three unsupervised optical flow regularizers
   • Rigid Motion Smoothness: L_smooth = Σ Charbonnier(∇u, ∇v)
   • Rotation Awareness: L_rotation = mean(clamp(div − curl, min=0)), penalizes divergence > curl
   • Flow Gradient Smoothness: L_gradient = Σ Charbonnier(∇²u, ∇²v)
6. Mathematical details of div/curl — discrete forward-difference approximation
7. Total loss composition — original losses + Kernal_Loss across 3 scales
8. Training engineering — AMP FP16, TF32, batch size 8→12, gradient clipping
9. Evaluation pipeline bug fix — DataParallel key prefix mismatch caused all epochs to score identically
10. Architecture comparison table (Original vs Deform vs HAHA)
11. Experimental setup — GoPro Large dataset, PSNR/SSIM metrics
12. Conclusions and future work
13. References

Each slide: 4–5 bullet points, academic style, include diagram suggestions.
```

---

## 【提示词 3】精简版（适合快速生成，发给 Kimi / 通义千问）

```
帮我生成一份 PPT 大纲（10 页），内容是：

**主题**：在 MISCFilter 图像去模糊论文基础上，做了两个改进：
1. 将标准卷积替换为可变形卷积（DCNv2），让网络自适应旋转模糊的形状
2. 在光流预测中加入三项物理先验约束（即 HAHA 版本）：
   - 刚性运动平滑：一阶光流梯度要小（∇u, ∇v）
   - 旋转感知：光流旋度应大于散度（curl > div）
   - 二阶梯度平滑：拉普拉斯光流梯度要小（∇²u, ∇²v）
3. 训练工程：AMP FP16 混合精度 + 梯度裁剪 + batch_size 提升
4. 修复了评估脚本中 DataParallel 权重加载的 Bug（key 前缀不匹配导致所有 epoch 得分相同）

**受众**：课题组汇报，有深度学习基础
**语言**：中文
**风格**：简洁学术，每页 3～5 条要点，建议配图类型
```

---

## 【提示词 4】专门针对 Gamma.app / Beautiful.ai 的结构化提示词

```
Generate a presentation with the following slides for an academic audience:

[SLIDE 1 - TITLE]
Title: MISCFilterNet-HAHA: Physical Prior Constraints for Rotational Blur Removal
Subtitle: Extending MISCFilter with Deformable Convolutions and Unsupervised Flow Regularization

[SLIDE 2 - PROBLEM]
Headline: The Challenge of Rotational Blur
• Motion blur from rotating objects (fans, turbine blades) is spatially non-uniform
• Standard optical flow methods produce noisy, physically inconsistent flow fields
• Existing deblurring networks lack physics-based supervision for rotation
Suggested visual: blurry fan image vs sharp image side-by-side

[SLIDE 3 - ORIGINAL MISCFILTER]
Headline: Original MISCFilter Architecture
• Multi-Input Single-Content Filter Network (MISC)
• 3-scale encoder-decoder with SCM, FAM, AFF modules
• Kernel prediction via optical flow warp at each scale
• Loss: Charbonnier + FFT frequency loss + edge loss
Suggested visual: network architecture diagram

[SLIDE 4 - DEFORMABLE CONVOLUTION]
Headline: Extension 1 — Deformable Convolutions
• Replace all standard convs with DCNv2 (learnable offsets)
• Modules: EBlock_Deform, DBlock_Deform, SCM_Deform, FAM_Deform, AFF_Deform
• ResBlock_Deform_fft_bench: FFT-based frequency features inside residual blocks
• Benefit: adaptive receptive fields that conform to motion trajectories
Suggested visual: deformable conv offset visualization

[SLIDE 5 - PHYSICAL CONSTRAINTS OVERVIEW]
Headline: Extension 2 — Three Physical Prior Constraints
• Constraint 1 — Rigid Smoothness: L = Σ√(|∇u|²+ε²) + √(|∇v|²+ε²)
• Constraint 2 — Rotation Awareness: L = mean(clamp(div−curl, min=0))
• Constraint 3 — 2nd-order Gradient Smoothness: L = Σ√(|∇²u|²+ε²) + √(|∇²v|²+ε²)
• Applied at all 3 scales; no center annotation needed; plug-and-play
Suggested visual: flow field showing curl vs div on rotating object

[SLIDE 6 - ROTATION AWARENESS MATH]
Headline: Physics of the Rotation-Awareness Loss
• curl = ∂v/∂x − ∂u/∂y (high curl = rotation)
• div = ∂u/∂x + ∂v/∂y (high div = expansion)
• Rotating rigid bodies: curl >> div, translation: curl ≈ 0
• Loss penalizes when div > curl: clamp(div−curl, min=0)
Suggested visual: vector field diagrams for pure rotation vs pure translation

[SLIDE 7 - TOTAL LOSS]
Headline: Combined Loss Function
• L_total = L_char + λ_fft·L_fft + λ_edge·L_edge + Kernal_Loss
• Kernal_Loss = Σ_scales [0.05·L_smooth + 0.03·L_rotation + 0.02·L_gradient]
• 9 constraint terms total (3 scales × 3 constraints)

[SLIDE 8 - TRAINING ENGINEERING]
Headline: Engineering Improvements
• AMP FP16: torch.amp.autocast + GradScaler → 1.5–2× speedup on RTX 4090
• TF32: torch.backends.cudnn.allow_tf32 = True (free speedup on Ampere)
• Batch size: 8 → 12 (dual-GPU, 6 per GPU)
• Gradient clipping: clip_grad_norm_(max_norm=0.5) prevents FP16 explosion
• NaN guard: skip corrupted batches, protect optimizer state

[SLIDE 9 - BUG FIX]
Headline: Evaluation Pipeline Bug Fix
• Bug: all 296 checkpoints scored identically (SSIM=0.0003, PSNR=5.01 dB)
• Root cause: DataParallel wrapping adds 'module.' prefix to state_dict keys;
  loader stripped prefix from saved keys but compared to model.state_dict()
  (also with prefix) → zero matches → empty weight load → random init every epoch
• Fix: use base_model = model.module (unwrapped) for key comparison
• Fix: switch evaluation to inference=True (single tensor output, no Kernal_Loss)
Suggested visual: code diff showing the 3-line fix

[SLIDE 10 - COMPARISON TABLE]
Headline: Architecture Comparison
(table: Original | Deform | HAHA)
Rows: Conv type | Physical constraints | FP16 training | Rotation awareness | Center annotation needed

[SLIDE 11 - EXPERIMENTS]
Headline: Experimental Setup
• Dataset: GoPro Large (2103 train pairs, 1111 test pairs)
• Metrics: PSNR (dB), SSIM, Score = 0.7×SSIM + 0.3×(PSNR/50)
• Training: 6000 epochs, Cosine Annealing LR, 3-epoch warmup
• Hardware: 2× NVIDIA GPU (RTX 4090 recommended)

[SLIDE 12 - CONCLUSIONS]
Headline: Contributions and Future Work
Contributions:
• Deformable convolutions for non-uniform blur handling
• Three unsupervised physics-based flow regularizers for rigid rotation
• AMP + TF32 training engineering for efficiency
• Evaluation pipeline bug fix (DataParallel key mismatch)
Future Work:
• Extension to video deblurring
• Real-world wind turbine blade dataset validation
• Ablation study on individual constraint weights
```

---

## 【各页配图建议汇总】

| 页码 | 推荐配图 |
|------|----------|
| 背景页 | 风机叶片模糊照片 vs 清晰照片 |
| MISCFilter 架构 | 三尺度 UNet 结构图（参考论文 Fig.1）|
| 可变形卷积 | DCNv2 偏移量可视化图（绿点偏移格网）|
| 物理约束 | 光流场向量图：旋转 vs 平移 vs 膨胀 |
| 散度/旋度 | 两个圆形向量场示意图（旋转 vs 辐射状）|
| 损失组成 | 损失权重饼图或加权求和公式图 |
| Bug 修复 | 修复前后代码对比（diff 高亮）|
| 对比表 | 三列对比表，绿色勾/红色叉 |
| 实验结果 | PSNR/SSIM 折线图（epoch vs 指标）|
