import torch
import torch.nn as nn
import torch.nn.functional as F

class KvCacheBatch(nn.Module):
    def __init__(self, max_blocks, block_size, kv_head_num, head_dim, device):
        super().__init__()
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.kv_head_num = kv_head_num
        self.head_dim = head_dim
        self.device = device

        self.kv_caches_dict = {}


    def get_kv_cache(self, layer_idx):
        if layer_idx not in self.kv_caches_dict:
            self.kv_caches_dict[layer_idx] = KvCache(
                self.max_blocks, self.block_size, self.kv_head_num, self.head_dim, self.device
            )
        return self.kv_caches_dict[layer_idx]


class KvCache:
    def __init__(self, max_blocks, block_size, kv_head_num, head_dim, device):
        super().__init__()


        self.kv_manager = KVCacheManager(
            max_blocks=max_blocks,
            block_size=block_size,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            device=device
        )

    def save(self, batch_seq_ids, k_batch, v_batch):
        batch_size = k_batch.shape[0]
        # 拆解 Batch，将当前的 k, v 写入散装物理池
        # TODO: Prefill 阶段的写入性能在长文本时会成为瓶颈。
        for b_idx in range(batch_size):
            seq_id = batch_seq_ids[b_idx]
            k_single = k_batch[b_idx:b_idx + 1]  # [1, num_kv_heads, seq_len, head_dim]
            v_single = v_batch[b_idx:b_idx + 1]

            # 写入池子 (零拷贝覆盖物理块)
            self.kv_manager.append_tokens(seq_id, k_single, v_single)

    def load(self, batch_seq_ids):
        batch_size = len(batch_seq_ids)

        # 从散装物理池中，重组成完整的、连续的张量喂给 FlashAttention (方案 A)
        # 注意：这里我们被迫要在 Python 层用 for 循环把 Batch 拼回来喂给标准的 F.scaled_dot_product_attention
        full_k_list = []
        full_v_list = []
        for b_idx in range(batch_size):
            seq_id = batch_seq_ids[b_idx]
            full_k, full_v = self.kv_manager.get_contiguous_kv(seq_id)  # 返回 [1, heads, total_seq_len, dim]
            full_k_list.append(full_k)
            full_v_list.append(full_v)

        # 拼回标准的 Batch 形状：[batch_size, num_kv_heads, max_total_seq_len, head_dim]
        # (这里如果不同句子长度不同，需要 padding 处理，为了简化，假设此时生成长度一致)
        k_contiguous = torch.cat(full_k_list, dim=0)
        v_contiguous = torch.cat(full_v_list, dim=0)

        return k_contiguous, v_contiguous

    # 从kv缓存获取当前词条的长度
    def get_seq_len_for_kv_cache(self, batch_seq_ids):
        lengths = []
        for seq_id in batch_seq_ids:
            lengths.append(self.kv_manager.seq_lengths.get(seq_id, 0))
        return lengths

# kv缓存管理器
class KVCacheManager:
    def __init__(self, max_blocks, block_size, kv_head_num, head_dim, device):
        super().__init__()

        """
            param max_blocks: 物理池中最多有多少个块 (例如 1024)
            param block_size: 每个块能装多少个 Token (通常是 16 或 32)
            param kv_head_num： kv有多少头
            param head_dim： qkv维度
        """

        self.block_size = block_size
        self.kv_head_num = kv_head_num
        self.head_dim = head_dim
        self.device = device

        # 🌟 1. 预先开辟物理显存池 (固定大小，不再动态申请)
        # 形状: [max_blocks, num_kv_heads, block_size, head_dim]
        self.k_pool = torch.zeros((max_blocks, kv_head_num, block_size, head_dim), dtype=torch.float16, device=device)
        self.v_pool = torch.zeros((max_blocks, kv_head_num, block_size, head_dim), dtype=torch.float16, device=device)

        # 🌟 2. 状态管理
        self.free_blocks = list(range(max_blocks))[::-1]  # 空闲块栈
        # 记录每个 sequence_id(每条文本)对应的块索引列表 -> block_table
        self.block_tables = {}  # sequence_id -> [block_id_1, block_id_2, ...]
        # 记录每个 sequence_id 目前总共有多少个 Token
        self.seq_lengths = {}

    def allocate_block(self):
        if not self.free_blocks:
            raise RuntimeError("Out of Memory! KV Cache Pool is full.")
        return self.free_blocks.pop()

    def init_sequence(self, seq_id):
        """为新的句子分配第一个block"""
        block_id = self.allocate_block()
        self.block_tables[seq_id] = [block_id]
        self.seq_lengths[seq_id] = 0

    def append_tokens(self, seq_id, k_states, v_states):
        """
        将新生成的 KV 写入物理池
        k_states: [1, num_kv_heads, seq_len, head_dim] (通常 seq_len=1)
        """
        # 🌟 修复 Bug：如果是第一次见到的 seq_id，先进行初始化！
        if seq_id not in self.seq_lengths:
            self.init_sequence(seq_id)

        seq_len = k_states.shape[2] # seq_len 代表输入字的长度

        for i in range(seq_len):
            curr_len = self.seq_lengths[seq_id] # 代表当前缓存字段的长度，还未加上当前输入的长度
            # 计算当前 Token 应该放在哪个块的哪个偏移位置
            block_idx = curr_len // self.block_size
            offset = curr_len % self.block_size

            # 如果当前块写满了，申请新块
            if block_idx >= len(self.block_tables[seq_id]):
                new_block_id = self.allocate_block()
                self.block_tables[seq_id].append(new_block_id)

            # 获取物理块的真实 ID
            physical_block_id = self.block_tables[seq_id][block_idx]

            # 🌟 写入物理池 (零拷贝覆盖)
            self.k_pool[physical_block_id, :, offset, :] = k_states[0, :, i, :]
            self.v_pool[physical_block_id, :, offset, :] = v_states[0, :, i, :]

            self.seq_lengths[seq_id] += 1

    def get_contiguous_kv(self, seq_id):
        """
        🌟 方案 A 的核心：从散装物理池中，重组成连续的张量喂给 FlashAttention
        返回形状: [1, num_kv_heads, total_seq_len, head_dim]
        """
        block_ids = self.block_tables[seq_id]
        total_len = self.seq_lengths[seq_id]

        # 把这个句子的所有物理块捞出来 [num_blocks_for_this_seq, num_kv_heads, block_size, head_dim]
        blocks_k = self.k_pool[block_ids]
        blocks_v = self.v_pool[block_ids]

        # 拼接并截断掉最后一个块中还没写满的无效部分
        # permute 后拉平: [num_kv_heads, num_blocks * block_size, head_dim]
        # 注意: 这违背了 Paged Attention 的极致性能初衷。真正的 Paged Attention 是连算子都在 Cuda 层重写的，计算注意力时直接拿着 Block Table 去物理池里东拼西凑地算内积，绝不在物理层做 cat。
        #   （备注：对于你的项目，目前保持现状即可，千万别去死磕写 C++ 算子，除非你想转行做推理系统底座。）
        k_contiguous = blocks_k.permute(1, 0, 2, 3).reshape(self.kv_head_num, -1, self.head_dim)
        v_contiguous = blocks_v.permute(1, 0, 2, 3).reshape(self.kv_head_num, -1, self.head_dim)

        # 增加 batch 维度并截断
        k_out = k_contiguous[:, :total_len, :].unsqueeze(0)
        v_out = v_contiguous[:, :total_len, :].unsqueeze(0)

        return k_out, v_out  # [1, num_kv_heads, total_seq_len, head_dim]