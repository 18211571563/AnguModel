from transformers import AutoTokenizer
import os

# ====================== 配置 ======================
HF_TOKEN = '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'
# 只下载 tokenizer（不需要模型）
TOKENIZER_NAME = "bert-base-chinese"  # 你可以换成任何：Llama/Qwen/GLM/Roberta 等
LOCAL_CACHE_DIR = "/home/georgy/model/tokenizer/hugingface_tokenizer/hugingface_tokenizer_cache"  # 本地缓存目录
LOCAL_SAVE_DIR  = "/home/georgy/model/tokenizer/hugingface_tokenizer"   # 最终保存到这里（干净独立）

# ====================== 1. 仅下载 Tokenizer（无模型） ======================
print("正在下载纯 Tokenizer...")

# 关键：trust_remote_code 按需开启
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    cache_dir=LOCAL_CACHE_DIR,
    force_download=False,
    resume_download=True,
    local_files_only=False,
    token = HF_TOKEN
)

# ====================== 2. 保存到本地（干净目录，训练用） ======================
tokenizer.save_pretrained(LOCAL_SAVE_DIR)
print(f"✅ 纯 Tokenizer 已保存到：{LOCAL_SAVE_DIR}")

# ====================== 3. 离线加载（断网可用，自己训练用） ======================
print("\n正在从本地离线加载 Tokenizer...")
my_tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_SAVE_DIR,
    local_files_only=True,  # 只看本地，不联网
    trust_remote_code=False,
)

# ====================== 测试 ======================
print("\n测试成功！")
print("Vocab 大小：", my_tokenizer.vocab_size)