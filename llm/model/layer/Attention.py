import torch
import torch.nn as nn
import torch.nn.functional as F
from llm.model.layer.RMSNorm import RMSNorm
from torch.nn.attention import sdpa_kernel, SDPBackend
from llm.model.layer.Rope import RotaryEmbedding
from llm.model.layer.KvCache import KvCache


class Attention(nn.Module):
    def __init__(self, dim, head_num, qk_norm = True, kv_group_num = 4):
        super().__init__()
        self.head_dim = dim // head_num     # 计算每头有多少dim
        self.head_num = head_num            # 有多少头
        self.kv_group_num = kv_group_num    # 一个q对应多少个KV
        self.kv_head_num = head_num // kv_group_num # kv有多少头

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_head_num * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_head_num * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

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
        if seq_len == 1:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                context = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # 走到这里，说明是多于 1 个 Token 一起进入网络 (比如首轮 Prompt 或多轮对话追加)
            kv_seq_len = k.shape[2]

            if seq_len == kv_seq_len:
                # 只有 Q 和 K 长度完全一样时，用内置的 is_causal 才最安全
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                # 💥 长度不一样！必须手写长方形 Mask 喂给 math 后端，FlashAttention 处理不了长方形的 causal
                mask = torch.full((seq_len, kv_seq_len), float('-inf'), device=x.device)
                mask = torch.triu(mask, diagonal=kv_seq_len - seq_len + 1)

                # ⚠️ 注意：带有自定义 Mask 时，很多老版本的 FlashAttention 不支持，会退化到 MATH
                with sdpa_kernel([SDPBackend.MATH]):
                    context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)



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



    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """🌟 修复 Bug 2：工业级 GQA 拉伸，零内存拷贝"""
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states.unsqueeze(2).expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)






