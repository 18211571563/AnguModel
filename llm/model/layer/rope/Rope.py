import torch
import torch.nn as nn
import torch.nn.functional as F
from llm.model.layer.rope.YaRNScaler import YaRNScaler

# ---------------------------------------------------------
# 全新封装】RoPE 位置编码类
# 优点：支持缓存管理、动态长度扩展，符合 2026 工业标准
# ---------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        # 预计算频率：theta^(-2i/dim) - 切换到 YaRN
        #inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        #self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 🌟 依赖注入：挂载 YaRN 策略引擎
        self.yarn_scaler = YaRNScaler(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            theta=theta
        )

        # 初始化缓存
        self._set_cos_sin_cache(max_position_embeddings, device=torch.device('cpu'))


    def forward(self, seq_len, device):
        # 如果当前需要的长度超过了缓存，则动态重新计算（2026 长文本外推核心）
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len, device)

        return (
            self.cos_cached[:seq_len, :].to(device),
            self.sin_cached[:seq_len, :].to(device)
        )

    """内部工厂函数：调用 YaRN 算力重新装填 cos/sin 缓存"""
    def _set_cos_sin_cache(self, seq_len, device):
        # 更新物理容量上限标记
        self.max_position_embeddings = seq_len

        # 1. 向 YaRN 索要针对此极长文本调校过的【频率分布】和【温度补偿】
        inv_freq_scaled, mscale = self.yarn_scaler.get_scaled_frequencies(seq_len, device)

        # 生成时间步 t = [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device).float()
        # 外积计算所有位置的频率角度
        # freqs = torch.outer(t, self.inv_freq) 切换到了 YaRN
        freqs = torch.outer(t, inv_freq_scaled)

        # 拼接成完整维度 [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # 🌟 YaRN: 将 YaRN 的温度补偿系数 (mscale) 直接注入 cos 和 sin
        cos_val = emb.cos() * mscale
        sin_val = emb.sin() * mscale

        # 注册为 buffer，不计入梯度，但随模型移动到 GPU/CPU
        self.register_buffer("cos_cached", cos_val, persistent=False)
        self.register_buffer("sin_cached", sin_val, persistent=False)



    # ---------------------------------------------------------
    # 1. 核心算子：旋转变换 (保持为辅助函数) - 转90度
    # ---------------------------------------------------------
    @staticmethod
    def rotate_half(x):
        """将向量的一半进行旋转，用于 RoPE 变换"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_emb(q, k, cos, sin):
        """应用旋转位置编码"""
        # 调整 cos/sin 形状以匹配 [batch, heads, seq, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rot = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_rot = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)
        return q_rot, k_rot