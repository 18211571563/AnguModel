import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.model.layer.KvCache import KvCacheBatch
from llm.model.layer.MoeSwiGlu import MoeSwiGlu
from llm.model.layer.RMSNorm import RMSNorm
from llm.model.layer.Attention import Attention
from llm.model.layer.Rope import RotaryEmbedding
from llm.common.config.ModelConfig import ModelConfig


class AnguModel(nn.Module):
    def __init__(self, config: ModelConfig):
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

        # weight tying: 让入口和出口用同一个矩阵
        # 好处：收敛的更快+节约显存
        # 不用weight tying理由：入口的初始维度信息和最后输出的维度信息，虽然大小一致，但是所代表的意义不一样，我觉得完全无法匹对
        #self.lm_head.weight = self.tok_embeddings.weight



    def forward(self, input_ids: torch.Tensor, kv_cache_batch:KvCacheBatch = None, batch_seq_ids = None, position_ids: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape

        # 🌟 如果外面没传 position_ids，我们就在这里算好！
        # 这样底层的网络就彻底和 KV Cache 长度解耦了
        if position_ids is None:
            past_seq_len = 0
            if kv_cache_batch is not None and batch_seq_ids is not None:
                past_seq_lens = kv_cache_batch.get_kv_cache(0).get_seq_len_for_kv_cache(batch_seq_ids)
                past_seq_len = past_seq_lens[0] if len(past_seq_lens) > 0 else 0
            
            # 生成标准的连续号码牌，例如 [10, 11, 12...]
            # 形状: [batch_size, seq_len]
            position_ids = torch.arange(
                past_seq_len, past_seq_len + seq_len, 
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        x = self.tok_embeddings(input_ids)
        x, total_aux_loss = self.swiGluLayer(x, kv_cache_batch, batch_seq_ids, position_ids)
        return self.lm_head(self.lm_norm(x)), total_aux_loss


class SwiGluLayer(nn.Module):
    def __init__(self, dim, hidden_dim, layer_num, head_num, expert_num, shared_num, top_k, max_seq_len):
        super().__init__()

        # 实例化 RoPE 类
        self.rope = RotaryEmbedding(dim // head_num, max_seq_len)

        self.layers = nn.ModuleList([
            SwiGluBlock(dim, hidden_dim, head_num, expert_num, shared_num, top_k) for _ in range(layer_num)
        ])

    def forward(self, x, kv_cache_batch:KvCacheBatch, batch_seq_ids, position_ids: torch.Tensor = None):
        batch, seq_len, dim = x.shape

        # ---------------------------------------------------------
        # 🌟 直接根据 position_ids 获取角度
        # ---------------------------------------------------------
        # 不再查询 KV Cache！我们直接看号码牌里最大的数字是多少
        max_pos = position_ids.max().item() + 1
        full_cos, full_sin = self.rope(max_pos) # [max_pos, head_dim/2]

        # 像查字典（Embedding）一样，直接用号码牌把对应的 cos 和 sin 拔出来！
        # 结果 cos 的形状是：[batch, seq_len, head_dim/2]
        cos = F.embedding(position_ids, full_cos)
        sin = F.embedding(position_ids, full_sin)

        '''
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
        '''

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


