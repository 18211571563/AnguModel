import math
import torch
import torch.nn as nn


class ParamInitializer:
    """
    负责模型权重的初始化与深度缩放。
    将其与模型核心前向逻辑解耦。
    """

    def __init__(self, num_layers: int, std: float = 0.02):
        self.num_layers = num_layers
        self.std = std
        # 预计算残差缩放系数
        self.res_scale = 1.0 / math.sqrt(2.0 * self.num_layers)

    def initialize(self, model: nn.Module):
        """
        统一执行初始化的入口函数
        """
        # 1. 基础正态分布初始化
        model.apply(self._init_weights)

        # 2. 残差分支深度感知缩放
        self._apply_depth_scaling(model)

        return model

    # ==========================================
    # 基础初始化：仅负责赋初始值
    # ==========================================
    def _init_weights(self, module: nn.Module):
        # 注意：这里不再需要传入 self 里面的模型，因为 apply 会自动把 module 传进来
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)

    # ==========================================
    # 深度感知缩放：负责把残差分支的输出压扁
    # ==========================================
    def _apply_depth_scaling(self, model: nn.Module):
        # 遍历模型的所有参数，根据命名路径进行精确打击
        for name, p in model.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                with torch.no_grad():
                    p.mul_(self.res_scale)