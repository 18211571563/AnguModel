import os
# 配置环境变量 - 7000xt系列显卡需要指定开 flashAttention
# 但因为对 RDNA3（gfx1101/1100）还在实验阶段，PyTorch 默认禁用它们，所以你看到这两条提示——它告诉你"这俩我能跑，但默认关了，要用就显式开"。
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # 指定使用

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from llm.model.AnguModel import AnguModel
from llm.common.TinyStoriesDataSetProcess import TinyStoriesDataSetProcess
from torch.utils.data import DataLoader
from tqdm import tqdm
from llm.common.config.Config import Config
import yaml



# ---------------------------------------------------------
# 1. 准备阶段 - 读取tokenizer
# ---------------------------------------------------------
local_tokenizer_path = r"/home/georgy/model/tokenizer/gpt-neox-20b"
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(local_tokenizer_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ---------------------------------------------------------
# 2. 准备阶段 - 配置
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = "../../resources/pth/my_mini_llm.pth"

# 加载模型配置
config_path = "../../resources/config/model_config.json"
config = Config.from_pretrained(config_path)

# 加载训练配置
with open("../../resources/config/train_config.yaml", "r") as f:
    train_cfg = yaml.safe_load(f)

pad_id = tokenizer.pad_token_id



# 🌟 训练超参数
local_data_path = train_cfg["data"]["local_data_path"] # 数据目录
max_length = train_cfg["data"]["max_length"]           # TinyStories 大部分故事都在 200 token 以内
load_data_size = train_cfg["data"]["load_data_size"]
epochs = train_cfg["training"]["epochs"]               # TinyStories 数据量极大，通常 1-2 个 Epoch 就能看效果
batch_size = train_cfg["training"]["batch_size"]       # 根据 7800XT 显存调整 (16G 显存可以尝试 16 或 32)
learning_rate = train_cfg["training"]["learning_rate"] # 预训练学习率可以稍微大一点点

print("learning_rate:", learning_rate)
print("epochs:", epochs)

# ---------------------------------------------------------
# 3. 准备训练数据
# ---------------------------------------------------------
'''
texts = [
    "人工智能是未来的发展趋势，掌握大模型技术非常重要。",
    "深度学习让计算机拥有了强大的泛化与认知能力。",
    "今天天气非常不错，我们一起去公园散步和放风筝吧。"
]
# 造数据阶段，直接给数据强制加上终止符：必须手动或通过代码，在结尾加上 eos_token
texts = [text + tokenizer.eos_token for text in texts]




inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
input_ids_full = inputs["input_ids"].to(device)

x = input_ids_full[:, :-1]      # x 是前 N-1 个词
y = input_ids_full[:, 1:]       # targets 是后 N-1 个词
'''
dataloader:DataLoader= TinyStoriesDataSetProcess.process(tokenizer,local_data_path = local_data_path, max_length=max_length, batch_size=batch_size, load_data_size=load_data_size)

# ---------------------------------------------------------
# 4. 加载模型阶段
# ---------------------------------------------------------
print("⏳ 加载模型阶段 - 开始...")
model = AnguModel(config).to(dtype=torch.float16, device=device)
if os.path.exists(model_save_path):
    print(f"🔄 加载模型阶段 - 发现预训练权重 '{model_save_path}'，正在加载...")
    model.load_state_dict(torch.load(model_save_path))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
print("✅ 加载模型阶段 - 模型加载完毕！")


# ---------------------------------------------------------
# 5. 实际训练阶段
# ---------------------------------------------------------
print(f"🔄 实际训练阶段 - 开始...")
for epoch in range(epochs):
    # 每次 Epoch 记录总 Loss
    epoch_loss = 0.0

    # 🌟 修改 1: tqdm 包装的是 enumerate(dataloader)
    # 并且传入 total=len(dataloader)，告诉 tqdm 总共有多少步
    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch + 1}/{epochs}",
        leave=True
    )

    for step, batch in progress_bar:
        # HuggingFace dataset 返回的是字典格式的 batch
        batch_ids = batch["input_ids"].to(device)
        # 错位标签逻辑
        x = batch_ids[:, :-1]
        y = batch_ids[:, 1:]

        optimizer.zero_grad()
        logits, aux_loss = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1), ignore_index=pad_id)
        loss = loss + aux_loss # 总 Loss = 主线 Loss + MoE 辅助罚款
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 🌟 每 10 步在进度条右侧动态更新一次最新的 Loss 和当前 Step
        if step % 10 == 0:
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}"
            })


    print(f"⭐ Epoch {epoch+1} 结束 | 平均 Loss: {(epoch_loss / len(dataloader)):.4f}\n")



# ---------------------------------------------------------
# 6. 保存模型
# ---------------------------------------------------------
torch.save(model.state_dict(), model_save_path)