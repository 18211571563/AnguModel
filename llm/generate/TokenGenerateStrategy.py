import torch
import torch.nn.functional as F

class TokenGenerateStrategy():
    def __init__(self):
        super().__init__()

    @staticmethod
    def process_temperature_and_top_p(logits:torch.Tensor, temperature, top_p):
        if temperature == 0:
            next_tokens = logits.argmax(dim=-1)  # shape: [batch_size]
        else:
            # 我们只需要最后一个字的打分
            # logits: [batch, vocab_size]
            next_token_logits = logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

                # 右移 -> 获取不满足条件的信息
                sorted_indices_to_remove = cumulative_probs > top_p  # 取出所有大于top_p的值
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # 右移
                sorted_indices_to_remove[..., 0] = 0  # 第一位设置为0

                # 把所有不满足条件的设置为0
                sorted_probs[sorted_indices_to_remove] = 0

                # 重新归一化 (让剩下的词概率加起来依然是 1)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # 🎲 4. 掷骰子：根据剩下的概率分布进行采样，选出 1 个下标
                next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)

                # 5. 把排序后的下标还原回词表的真实 ID -> 这个next_token_sorted_idx是重新排序后的，如果直接返回会错乱，要找回排序前的id
                next_tokens = torch.gather(sorted_indices, -1, next_token_sorted_idx)

            else:
                next_tokens = torch.multinomial(probs, num_samples=1)

            # 压缩维度
            next_tokens = next_tokens.squeeze(-1)

        return next_tokens