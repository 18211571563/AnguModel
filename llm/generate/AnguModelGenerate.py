import os
import torch
from transformers import AutoTokenizer
#from transformers import BertTokenizer
from transformers import GPTNeoXTokenizer
from llm.model.AnguModel import AnguModel
from llm.generate.TokenGenerateStrategy import TokenGenerateStrategy
from llm.common.config.ModelConfig import ModelConfig
from llm.model.layer.KvCache import KvCacheBatch
import yaml


# ---------------------------------------------------------
# 1. 准备阶段 - 配置
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 🌟 加载模型配置
model_config:ModelConfig = ModelConfig.from_pretrained("../../resources/config/model_config.json")

# 🌟 加载训练配置
with open("../../resources/config/train_config.yaml", "r") as f:
    train_cfg = yaml.safe_load(f)

model_save_path = train_cfg["pth"]["model_save_path"]
temperature = train_cfg["generate"]["temperature"]
top_p = train_cfg["generate"]["top_p"]

# ---------------------------------------------------------
# 2. 准备阶段 - tokenizer
# ---------------------------------------------------------
local_tokenizer_path = r"/home/georgy/model/tokenizer/gpt-neox-20b"
tokenizer:GPTNeoXTokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
print(type(tokenizer))

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

# ---------------------------------------------------------
# 3. 准备训练数据
# ---------------------------------------------------------
texts = [
    "hello，",
    "mysql is",
    "my name"
]

inputs = tokenizer(texts, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device)

# 【BPE 修改】: 因果语言模型不需要切掉结尾。输入的 prompt 是多少，就送进去多少。
x = input_ids
#x = input_ids[:, :-1]      # x 是前 N-1 个词, 去掉结尾的 [SEP]

# 1. 基础设置
# ⚠️ 注意：虽然能自动结束，但依然必须设置一个 max_generate_length！
# 这是为了防止模型“发疯”（比如陷入死循环一直不输出 [SEP]），这叫“安全刹车”。
max_generate_length = 100

pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id
print("pad_token_id:", pad_token_id)
print("eos_token_id:", eos_token_id)

current_token_id = x
generate_token_id = x

# kv cache
kv_cache_batch = KvCacheBatch(1024, 16, model_config.head_num // 4, model_config.dim // model_config.head_num, device)
batch_size = input_ids.shape[0]
batch_seq_ids = [1000 + i for i in range(batch_size)] # 比如 [1000, 1001, 1002...]

batch_size = current_token_id.shape[0]
unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device) # 定义每行初始化都是 True

# =========================================================
# 🌟 核心准备：记录初始 Prompt 的真实长度，用于后续推算号码牌
# =========================================================
initial_seq_len = current_token_id.shape[1]

# ---------------------------------------------------------
# 4. 加载模型阶段
# ---------------------------------------------------------
print("⏳ 加载模型阶段 - 开始...")
model = AnguModel(model_config).to(dtype=torch.bfloat16, device=device)
if os.path.exists(model_save_path):
    print(f"🔄 加载模型阶段 - 发现预训练权重 '{model_save_path}'，正在加载...")
    model.load_state_dict(torch.load(model_save_path))
else:
    raise Exception(f"没有找到模型, 模型地址: '{model_save_path}'")

model.eval()
print("✅ 加载模型阶段 - 模型加载完毕！")


# ---------------------------------------------------------
# 5. 实际训练阶段
# ---------------------------------------------------------
for step in range(61):
    with torch.no_grad():
        # ---------------------------------------------------------
        # 🌟 新增：动态计算当前轮次的 position_ids
        # ---------------------------------------------------------
        if step == 0:
            # 第一次循环 (Prefill)：处理完整的 prompt
            # 给每个 token 发 0 到 initial_seq_len-1 的绝对位置号码牌
            position_ids = torch.arange(
                0, initial_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            # 第二次及以后的循环 (Decode)：每次只处理刚生成的 1 个新 token
            # 这 1 个 token 的绝对位置就是：初始长度 + 已经走过的步数 - 1
            current_pos = initial_seq_len + step - 1
            position_ids = torch.full(
                (batch_size, 1), current_pos, dtype=torch.long, device=device
            )

        logits, _ = model(generate_token_id, kv_cache_batch, batch_seq_ids, position_ids)  # [batch, seq_len, vocab_size]
        last_step_logits = logits[:, -1, :]                 # [batch, vocab_size]

        # token处理 -> 主要基于模型计算出来的每行对应的下个词的概率信息，如何取下个词
        next_tokens = TokenGenerateStrategy.process_temperature_and_top_p(last_step_logits, temperature, top_p)

        # 连续生成长句子处理
        # TODO: 添加对空列表的处理逻辑 要试试没有这句效果
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (~unfinished_sequences) # 注意:如果语句结束,直接补充上 PAD, 否则用回生成的词

        # 【BPE 修改】: 判断结束的条件变为 eos_token_id
        is_done = (next_tokens == eos_token_id)
        # is_done = (next_tokens == sep_token_id)

        unfinished_sequences = unfinished_sequences & ~is_done # bool计算: 矩阵的 &

        # 记录当前词句和本次生成的单个词
        current_token_id = torch.cat([current_token_id, next_tokens.unsqueeze(-1)], 1)
        generate_token_id = next_tokens.unsqueeze(-1)

        # 每行都有结束符号，退出 -> 全部False代表结束
        if not unfinished_sequences.any():
            break

print("\n🎉 预测结果：")
for i in range(len(texts)):
    # 【BPE 修改】: 建议加上 skip_special_tokens=True，这样打印出来就不会有一堆尾部补齐的标志了
    next_word = tokenizer.decode(current_token_id[i], skip_special_tokens=True)
    # next_word = tokenizer.decode(current_token_id[i])
    print(f"人类输入: '{texts[i]}'  =>  AI 预测: '{next_word}'")





