from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoeSwiGlu(nn.Module):
    def __init__(self, dim, hidden_dim, expert_num, shared_num, top_k):
        super().__init__()
        self.expert_num = expert_num
        self.top_k = top_k
        self.expert_gateway = nn.Linear(dim, expert_num, bias=False)
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(expert_num)])

        if shared_num > 0:
            self.shared_experts = Expert(dim, hidden_dim * shared_num)
            # 增加一个控制共享专家能量的门控或标量（初始化为 1.0）
            self.shared_gate = nn.Parameter(torch.ones(1))
        else:
            self.shared_experts = None


    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x = x.view(-1, dim)

        # ==========================================================
        # 二, 共享专家
        # ==========================================================
        if self.shared_experts is not None:
            shared_output = self.shared_experts(x) * self.shared_gate
        else:
            shared_output = 0


        # ==========================================================
        # 二, 路由专家: 全局 Softmax -> 挑出 Top-K 的概率 -> 把这 K 个概率除以它们的和 (重归一化)
        # ==========================================================
        router_logits = self.expert_gateway(x)  # [batch * seq_len, expert_num]  ps: 可能含有负数
        router_logits_softmax = F.softmax(router_logits, dim=1)  # [batch * seq_len, expert_num] ps: 经过softmax后不会出现负数
        routing_weights, selected_experts = torch.topk(router_logits_softmax, k=self.top_k, dim=1) # [batch_size * seq_len, weight]
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True) # [batch_size * seq_len, top_k]

        # ==========================================================
        # 三, 专家路由Loss: 🌟 2026 前沿升级 MoE 负载均衡 (Balancing Loss) & Z-Loss
        # ==========================================================
        if self.training:  # 只有在训练模式下才算这玩意儿
            # 1. 计算 Balancing Loss (防止专家饿死)
            # P_i: 网关给每个专家的预测概率均值
            p_i = router_logits_softmax.mean(dim=0)  # [expert_num] expert_num个庄家的平均数

            # f_i: 每个专家实际被选中 (Top-K) 的比例  = 每个专家被选中的次数/所有专家被选中的次数
            # 用 one_hot 搞出形状为 [batch*seq_len, top_k, expert_num] 的张量
            mask = F.one_hot(selected_experts, num_classes=self.expert_num).float()
            # 在 batch 维度求均值，得到真实被分配的频率
            f_i = mask.sum(dim=1).mean(dim=0) / self.top_k  # 🌟 关键：除以 top_k 归一化
            # 公式：expert_num * sum(P_i * f_i)
            # alpha 系数设为 0.01 (可调超参)
            bal_loss = torch.sum(p_i * f_i) * self.expert_num * 0.01

            # 1. 计算 z_loss (惩罚值最大的，防止爆炸)
            # 每个词对于的8个专家中最高得分的那个，之后平方后取平均值 * 0.0001
            # 2. 深入张量分析：
                #1.  `logsumexp` 的数学设计初衷是作用在 **原始未归一化的 logits（通常有正有负，甚至会飙到几十）** 上的。
                #2.  它的公式是 $\log(\sum e^{x_i})$。
                #3.  因为你在前面已经执行了 `router_logits = F.softmax(router_logits, dim=1)`，所以传进 `logsumexp` 的 `x_i` 全部变成了 `0` 到 `1` 之间的概率值。
                #4.  如果你对一组概率值（和为 1）求 `logsumexp`，它算出来的东西**在数学上失去了惩罚极端大值的意义**。
            z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2) * 0.0001

            # 总结：防爆炸 -> 两者的分工
            #   这就好比：
            #       残差和 Norm 是物理隔离带，防止整片森林（整个大模型）起火；
            #       Z-Loss 是针对某个喜欢玩火的小孩（MoE网关）的家教，你不能没收他的打火机（因为他要用来生火做饭），只能每次他火烧得太大时，打他的屁股。
            #
            #   x = x + f(norm(x))（架构级防护）：
            #       解决的是多层叠加引起的系统性膨胀。它保证了信号在纵向穿越 100 层网络时，还能保持健康的体态。
            #   Z-Loss（损失函数级防护）：
            #       解决的是MoE 网关局部的极端狂热。我们不能在架构上限制网关（因为它需要拉开分数差距来做决策），所以我们保留它打高分的权利，但通过 Z-Loss 这种“经济罚款”的方式，在背后警告它：“你可以有差距，但不能大得离谱”。
            aux_loss = bal_loss + z_loss

        else:
            # 推理时不计算，直接给 0
            aux_loss = 0.0

        # ==========================================================
        # 四, 普通专家 = 专家计算结果 * 权重
        # ==========================================================
        # 准备一个全零张量来接收最终结果
        final_hidden_states = torch.zeros_like(x)
        for expert_idx in range(self.expert_num):
            # 选择专家: 基于路由专家的结果进行选择
            token_idx_1, token_idx_2 = torch.where(selected_experts == expert_idx) # [batch_size * seq_len, top_k]
            if token_idx_1.shape[0] == 0:
                continue

            expert_layer = self.experts[expert_idx] # 从列表取出专家
            current_x = x[token_idx_1] # [token_idx, dim] 取出匹配中此专家的行seq
            current_hidden_x = expert_layer(current_x)  # 结果: 送入专家进行计算(核心计算在这里发生) [token_idx, dim]

            # 乘上路由器的权重 (越受路由器青睐，权重越大)
            current_routing_weights = routing_weights[token_idx_1, token_idx_2].unsqueeze(-1)
            # 这2个可以相乘，是因为行数相同，行数是从 selected_experts == expert_idx 算出 token_idx_1
            current_hidden_states = current_hidden_x * current_routing_weights  # [token_idx, dim]

            # 用于沿指定行方向维度，将 current_hidden_states 张量的值按 token_idx 索引规则累加到原张量中
            final_hidden_states.index_add_(0, token_idx_1, current_hidden_states)

        final_hidden_states = final_hidden_states + shared_output
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, dim)
        return final_hidden_states, aux_loss


class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))