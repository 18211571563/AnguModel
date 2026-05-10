import torch
import torch.nn as nn
import torch.nn.functional as F
from llm.model.modules.RMSNorm import RMSNorm
from torch.nn.attention import sdpa_kernel, SDPBackend
from llm.model.modules.rope.Rope import RotaryEmbedding
from llm.model.modules.KvCache import KvCache
from llm.model.config.ModelConfig import ModelConfig
import math


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head_dim = config.dim // config.head_num     # 计算每头有多少dim
        self.head_num = config.head_num            # 有多少头
        self.kv_group_num = config.kv_group_num    # 一个q对应多少个KV
        self.kv_head_num = config.head_num // config.kv_group_num # kv有多少头

        # 核心：根据层号决定这层是 Global 还是 SWA
        # 比如：偶数层是全局，奇数层是 SWA
        self.is_swa = (config.layer_num % 2 != 0)
        self.window_size = config.sliding_window_size if self.is_swa else None

        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.kv_head_num * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.kv_head_num * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()

    def forward(self, x, cos, sin, kv_cache:KvCache = None, batch_seq_ids=None):
        batch_size, seq_len, dim = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.kv_head_num, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.kv_head_num, self.head_dim).transpose(1, 2)

        # ---------------------------------------------------------
        # 🌟 QK-Norm 🌟
        # ---------------------------------------------------------
        q, k = self.q_norm(q), self.k_norm(k)

        # ---------------------------------------------------------
        # 🌟 RoPE 🌟
        # ---------------------------------------------------------
        q, k = RotaryEmbedding.apply_rotary_emb(q, k, cos, sin)

        # ---------------------------------------------------------
        # 🌟 KV-Cache 🌟
        # ---------------------------------------------------------
        if kv_cache is not None:
            kv_cache.save(batch_seq_ids, k, v)
            k, v = kv_cache.load(batch_seq_ids)

        # 🌟 GQA 广播魔法：把 2 个头的 K/V 复制拉伸成 8 个头，去迎合 Q
        # 重复次数 = 4 = heads_num // num_kv_heads
        k = Attention.repeat_kv(k, self.kv_group_num)
        v = Attention.repeat_kv(v, self.kv_group_num)

        # flashAttention
        scale_factor = 1.0 if isinstance(self.q_norm, RMSNorm) else (1.0 / math.sqrt(self.head_dim))

        if seq_len == 1:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                context = F.scaled_dot_product_attention(q, k, v, scale=scale_factor, is_causal=False)
        else:
            # 走到这里，说明是多于 1 个 Token 一起进入网络 (比如首轮 Prompt 或多轮对话追加)
            kv_seq_len = k.shape[2]

            if seq_len == kv_seq_len:
                # 只有 Q 和 K 长度完全一样时，用内置的 is_causal 才最安全
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    context = F.scaled_dot_product_attention(q, k, v, scale=scale_factor, is_causal=True)
            else:
                # 💥 长度不一样！必须手写长方形 Mask 喂给 math 后端，FlashAttention 处理不了长方形的 causal
                mask = self._create_attention_mask(seq_len, kv_seq_len, x.device)

                # ⚠️ 注意：带有自定义 Mask 时，很多老版本的 FlashAttention 不支持，会退化到 MATH
                with sdpa_kernel([SDPBackend.MATH]):
                    context = F.scaled_dot_product_attention(q, k, v, scale=scale_factor, attn_mask=mask)



        ''' 
            # 此代码替换成 flashAttention

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)


            # 由于kv cache 会出现 既有 past_kv，当前 seq_len 又大于 1 的情况，下面代码不可用
            # if seq_len > 1:
            #     mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
            #     scores = scores + mask

            # 由于kv cache 会出现 既有 past_kv，当前 seq_len 又大于 1 的情况， 改造
            if seq_len > 1:
                # 🌟 修复 Bug 2：动态获取 K 的真实长度，生成长方形 Mask
                kv_seq_len = k.shape[2]
                mask = torch.full((seq_len, kv_seq_len), float('-inf'), device=x.device)
                # 遮挡住右上角的未来信息
                mask = torch.triu(mask, diagonal=kv_seq_len - seq_len + 1)
                scores = scores + mask

            attn = F.softmax(scores, dim=-1)
            context = torch.matmul(attn, v)
        '''

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.o_proj(context)


    def _create_attention_mask(self, seq_len, kv_seq_len, device):
        mask = torch.full((seq_len, kv_seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=kv_seq_len - seq_len + 1)

        # swa 正方形才这样处理，不过不合适了
        #if self.is_swa and self.window_size is not None:
        #    swa_mask = torch.tril(torch.ones(seq_len, kv_seq_len, device=device), diagonal=-self.window_size).bool()
        #    mask.masked_fill_(swa_mask, float('-inf'))

        # 2. 长方形下绝对安全的 SWA Mask
        if self.is_swa and self.window_size is not None:
            # 建立 Q 和 K 的绝对物理位置索引
            # K 的位置: [0, 1, 2, ..., kv_seq_len-1]
            k_positions = torch.arange(kv_seq_len, device=device).view(1, -1)
            # Q 的位置: [kv_seq_len - seq_len, ..., kv_seq_len-1]
            # 把它变成列向量，方便广播相减
            q_positions = torch.arange(kv_seq_len - seq_len, kv_seq_len, device=device).view(-1, 1)
            # 距离 = 当前词位置 - 历史词位置
            distance = q_positions - k_positions
            # 当距离大于等于窗口大小，就是远古历史，必须屏蔽！
            swa_mask = distance >= self.window_size
            mask.masked_fill_(swa_mask, float('-inf'))

        return mask


    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """🌟 修复 Bug 2：工业级 GQA 拉伸，零内存拷贝"""
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states.unsqueeze(2).expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)






