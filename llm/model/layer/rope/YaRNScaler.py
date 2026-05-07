import torch
import math

class YaRNScaler:
    """
    YaRN 频率缩放策略封装模块 (完全解耦)
    职责：根据序列长度动态计算修正后的 RoPE 频率 (inv_freq) 和温度系数 (mscale)
    """
    def __init__(self, dim, max_position_embeddings=4096, theta=10000.0, beta_fast=32, beta_slow=1):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        # [状态隔离] 提前算好基础参数，避免每次计算时重复开销
        self.dim_indices = torch.arange(0, self.dim, 2, dtype=torch.float32)
        self.inv_freq_base = 1.0 / (self.theta ** (self.dim_indices / self.dim))
        self.wavelengths = 2 * math.pi / self.inv_freq_base
        
        # 预计算好高低频的物理界限 (避免在 forward 里算乘法)
        self.low_freq_bound = self.beta_slow * 2 * math.pi
        self.high_freq_bound = self.beta_fast * 2 * math.pi

    def _get_mscale(self, scale):
        """计算解决注意力熵崩塌的温度系数"""
        if scale <= 1.0:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def get_scaled_frequencies(self, seq_len, device):
        """核心对外交口：传入长度，返回加工好的频率和系数"""
        scale = max(1.0, seq_len / self.max_position_embeddings)
        
        # 转移到目标设备
        inv_freq_base = self.inv_freq_base.to(device)
        wavelengths = self.wavelengths.to(device)

        # 1. 文本在安全范围内，直接返回原始频率，系数为 1
        if scale == 1.0:
            return inv_freq_base, 1.0

        # 2. 文本超载，执行 YaRN 三段式计算
        # 计算 gamma (限制在 0~1 之间)
        gamma = (wavelengths - self.high_freq_bound) / (self.low_freq_bound - self.high_freq_bound)
        gamma = torch.clamp(gamma, min=0.0, max=1.0)
        
        # 混合频率：高频原样保留(gamma=0)，低频线性缩小(gamma=1)
        inv_freq_scaled = inv_freq_base * (1 - gamma) + (inv_freq_base / scale) * gamma
        
        # 算温度系数
        mscale = self._get_mscale(scale)
        
        return inv_freq_scaled, mscale