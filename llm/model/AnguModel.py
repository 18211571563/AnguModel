import torch
import torch.nn as nn

from llm.model.layer.KvCache import KvCache
from llm.model.layer.KvCache import KvCacheBatch
from llm.model.layer.MoeSwiGlu import MoeSwiGlu
from llm.model.layer.RMSNorm import RMSNorm
from llm.model.layer.Attention import Attention
from llm.model.layer.Rope import RotaryEmbedding
from llm.common.config.Config import Config


class AnguModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)

        self.swiGluLayer = SwiGluLayer(
            self.config.dim,
            self.config.hidden_dim,
            self.config.layer_num,
            self.config.head_num,
            self.config.expert_num,
            self.config.shared_num,
            self.config.top_k,
            self.config.max_seq_len
        )

        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        self.lm_norm = RMSNorm(self.config.dim)


    def forward(self, input_ids: torch.Tensor, kv_cache_batch:KvCacheBatch = None, batch_seq_ids = None):
        x = self.tok_embeddings(input_ids)
        x, total_aux_loss = self.swiGluLayer(x, kv_cache_batch, batch_seq_ids)
        return self.lm_head(self.lm_norm(x)), total_aux_loss


class SwiGluLayer(nn.Module):
    def __init__(self, dim, hidden_dim, layer_num, head_num, expert_num, shared_num, top_k, max_seq_len):
        super().__init__()

        # 实例化 RoPE 类
        self.rope = RotaryEmbedding(dim // head_num, max_seq_len)

        self.layers = nn.ModuleList([
            SwiGluBlock(dim, hidden_dim, head_num, expert_num, shared_num, top_k) for _ in range(layer_num)
        ])

    def forward(self, x, kv_cache_batch:KvCacheBatch, batch_seq_ids):
        batch, seq_len, dim = x.shape

        # ---------------------------------------------------------
        # 🌟 GQA 🌟 从类中动态获取 cos 和 sin -> 使用kv cache的代码，导致seq_len默认都是1，需要加上历史长度
        # ---------------------------------------------------------
        #  从类中动态获取 cos 和 sin -> 使用kv cache的代码，导致seq_len默认都是1，需要加上历史长度
        past_seq_len = 0
        if kv_cache_batch is not None and batch_seq_ids is not None:
            # 🌟 取列表中的第一个元素（默认 batch 内长度是对齐的）
            past_seq_lens = kv_cache_batch.get_kv_cache(0).get_seq_len_for_kv_cache(batch_seq_ids)
            past_seq_len = past_seq_lens[0] if len(past_seq_lens) > 0 else 0
        total_seq_len = past_seq_len + seq_len
        full_cos, full_sin = self.rope(total_seq_len)

        # ！！！仅仅切片出【当前正在输入的这几个字】对应的角度 ！！！
        cos = full_cos[past_seq_len: total_seq_len]
        sin = full_sin[past_seq_len: total_seq_len]

        # Meo： 网关损失函数对外传递
        total_aux_loss = 0.0  # 🌟 准备一个总账本
        # 叠层
        for i, layer in enumerate(self.layers):
            kv_cache = kv_cache_batch.get_kv_cache(i) if kv_cache_batch is not None else None
            x, aux_loss = layer(x, cos, sin, kv_cache, batch_seq_ids)

            if self.training:
                total_aux_loss += aux_loss

        return x, total_aux_loss


class SwiGluBlock(nn.Module):
    def __init__(self, dim, hidden_dim, head_num, expert_num, shared_num, top_k):
        super().__init__()
        self.norm_attention = RMSNorm(dim)
        self.self_attention = Attention(dim, head_num)
        self.norm = RMSNorm(dim)
        self.swig = MoeSwiGlu(dim, hidden_dim, expert_num, shared_num, top_k)

    def forward(self, x, cos, sin, kv_cache, batch_seq_ids):
        x_attention = self.self_attention(self.norm_attention(x), cos, sin, kv_cache, batch_seq_ids)
        x = x + x_attention
        x_swig, aux_loss = self.swig(self.norm(x))
        x = x + x_swig
        return x, aux_loss


