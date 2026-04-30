import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 全新封装】RoPE 位置编码类
# 优点：支持缓存管理、动态长度扩展，符合 2026 工业标准
# ---------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率：theta^(-2i/dim)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 初始化缓存
        self._set_cos_sin_cache(max_seq_len)


    def forward(self, seq_len):
        # 如果当前需要的长度超过了缓存，则动态重新计算（2026 长文本外推核心）
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len, :],
            self.sin_cached[:seq_len, :]
        )


    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len = seq_len
        # 生成时间步 t = [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        # 外积计算所有位置的频率角度
        freqs = torch.outer(t, self.inv_freq)
        # 拼接成完整维度 [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # 注册为 buffer，不计入梯度，但随模型移动到 GPU/CPU
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)



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
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        q_rot = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_rot = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)
        return q_rot, k_rot