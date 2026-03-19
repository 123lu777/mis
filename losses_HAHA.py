"""
光流物理先验约束损失函数
Physical Prior Constraint Loss Functions for Optical Flow

针对刚性旋转运动（如风机叶片）设计，三个约束：
1. RigidMotionSmoothnessLoss  - 刚性运动平滑约束（一阶梯度）
2. RotationAwarenessLoss      - 无中心旋转感知约束（curl > divergence）
3. FlowGradientSmoothnessLoss - 流梯度二阶平滑约束

核心优势：
- 不需要旋转中心标注
- 不需要分割掩码
- 整图统一约束，即插即用
"""

import torch
import torch.nn as nn


class RigidMotionSmoothnessLoss(nn.Module):
    """
    刚性运动平滑约束

    物理公式：L_smooth = Σ(|∇u| + |∇v|)
    其中 u, v 分别是光流场的 x 和 y 方向分量。

    物理含义：
    - 刚性物体不会扭曲变形，相邻像素的光流差异应该很小
    - 使用 Charbonnier 损失（平滑L1），对异常值鲁棒
    - 约束光流梯度的一阶导数，保证局部运动一致性
    """

    def __init__(self, epsilon=0.001):
        super(RigidMotionSmoothnessLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, flow):
        """
        Args:
            flow: 光流场，形状 [B, 2, H, W]
                  flow[:, 0] 为 u（x方向），flow[:, 1] 为 v（y方向）
        Returns:
            loss: 标量，刚性平滑损失
        """
        u = flow[:, 0:1, :, :]  # x方向光流 [B, 1, H, W]
        v = flow[:, 1:2, :, :]  # y方向光流 [B, 1, H, W]

        # 一阶梯度：前向差分（与 RotationAwarenessLoss 方向一致）
        du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]   # ∂u/∂x
        du_dy = u[:, :, 1:, :] - u[:, :, :-1, :]   # ∂u/∂y
        dv_dx = v[:, :, :, 1:] - v[:, :, :, :-1]   # ∂v/∂x
        dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]   # ∂v/∂y

        # Charbonnier 损失：sqrt(x^2 + ε^2)，比 |x| 更平滑、对梯度更友好
        loss = (torch.mean(torch.sqrt(du_dx ** 2 + self.epsilon ** 2)) +
                torch.mean(torch.sqrt(du_dy ** 2 + self.epsilon ** 2)) +
                torch.mean(torch.sqrt(dv_dx ** 2 + self.epsilon ** 2)) +
                torch.mean(torch.sqrt(dv_dy ** 2 + self.epsilon ** 2)))
        return loss


class RotationAwarenessLoss(nn.Module):
    """
    无中心旋转感知约束（curl > divergence）

    物理公式：L_rotation = mean(max(0, div - curl))
    其中：
        curl = ∂v/∂x - ∂u/∂y  （旋度：越大越像旋转）
        div  = ∂u/∂x + ∂v/∂y  （散度：越大越像膨胀/收缩）

    物理含义：
    - 旋转运动特征：curl 大、div 小（转圈不膨胀）
    - 平移运动特征：curl 小
    - 膨胀/收缩特征：div 大
    - 惩罚 div > curl 的情况，鼓励光流场呈现旋转特性
    - 完全不需要知道旋转中心！
    """

    def forward(self, flow):
        """
        Args:
            flow: 光流场，形状 [B, 2, H, W]
        Returns:
            loss: 标量，旋转感知损失
        """
        u = flow[:, 0:1, :, :]  # x方向光流 [B, 1, H, W]
        v = flow[:, 1:2, :, :]  # y方向光流 [B, 1, H, W]

        # 计算偏导数（前向差分，长度 H-1 或 W-1）
        du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]   # ∂u/∂x  [B, 1, H, W-1]
        du_dy = u[:, :, 1:, :] - u[:, :, :-1, :]   # ∂u/∂y  [B, 1, H-1, W]
        dv_dx = v[:, :, :, 1:] - v[:, :, :, :-1]   # ∂v/∂x  [B, 1, H, W-1]
        dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]   # ∂v/∂y  [B, 1, H-1, W]

        # 取公共区域（去掉最后一行/列以对齐尺寸）
        curl = dv_dx[:, :, :-1, :] - du_dy[:, :, :, :-1]  # ∂v/∂x - ∂u/∂y
        div  = du_dx[:, :, :-1, :] + dv_dy[:, :, :, :-1]  # ∂u/∂x + ∂v/∂y

        # 惩罚 div > curl 的情况（max(0, div - curl)）
        # 意味着：如果散开程度 > 旋转程度，就施加惩罚
        loss = torch.mean(torch.clamp(div - curl, min=0.0))
        return loss


class FlowGradientSmoothnessLoss(nn.Module):
    """
    流梯度二阶平滑约束

    物理公式：L_gradient = Σ(|∇²u| + |∇²v|)
    其中 ∇² 是拉普拉斯算子（二阶导数）。

    物理含义：
    - 不仅光流本身要平滑，光流的变化率也要平滑
    - 防止光流场出现"碎裂"、"抖动"现象
    - 比一阶平滑约束更强，保证从粗到细的多尺度连续性
    """

    def __init__(self, epsilon=0.001):
        super(FlowGradientSmoothnessLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, flow):
        """
        Args:
            flow: 光流场，形状 [B, 2, H, W]
        Returns:
            loss: 标量，二阶梯度平滑损失
        """
        u = flow[:, 0:1, :, :]  # x方向光流 [B, 1, H, W]
        v = flow[:, 1:2, :, :]  # y方向光流 [B, 1, H, W]

        # 二阶导数（拉普拉斯离散近似）：f[x+1] - 2*f[x] + f[x-1]
        d2u_dx2 = u[:, :, :, 2:] - 2 * u[:, :, :, 1:-1] + u[:, :, :, :-2]  # ∂²u/∂x²
        d2u_dy2 = u[:, :, 2:, :] - 2 * u[:, :, 1:-1, :] + u[:, :, :-2, :]  # ∂²u/∂y²
        d2v_dx2 = v[:, :, :, 2:] - 2 * v[:, :, :, 1:-1] + v[:, :, :, :-2]  # ∂²v/∂x²
        d2v_dy2 = v[:, :, 2:, :] - 2 * v[:, :, 1:-1, :] + v[:, :, :-2, :]  # ∂²v/∂y²

        # Charbonnier 损失
        loss = (torch.mean(torch.sqrt(d2u_dx2 ** 2 + self.epsilon ** 2)) +
                torch.mean(torch.sqrt(d2u_dy2 ** 2 + self.epsilon ** 2)) +
                torch.mean(torch.sqrt(d2v_dx2 ** 2 + self.epsilon ** 2)) +
                torch.mean(torch.sqrt(d2v_dy2 ** 2 + self.epsilon ** 2)))
        return loss
