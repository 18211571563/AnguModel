import os
import sys

# 1. 强制注入国内镜像源环境变量（必须放在所有 HF 库导入之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用软链接，强制下载真实文件到指定目录
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from transformers import AutoTokenizer
except ImportError:
    print("❌ 缺少 transformers 库，请先执行: pip install transformers")
    sys.exit(1)

# ==========================================
# ⚙️ 配置区域
# ==========================================
MODEL_ID = "EleutherAI/gpt-neox-20b"
SAVE_DIR = "/home/georgy/model/tokenizer/gpt-neox-20b"


def download_and_save_tokenizer():
    print(f"🚀 正在通过镜像源请求 {MODEL_ID} 的 Tokenizer 文件...")

    try:
        # 确保保存目录存在
        os.makedirs(SAVE_DIR, exist_ok=True)

        # 从 HuggingFace 加载 Tokenizer
        print("   正在下载配置和词表文件 (通常小于 5MB)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # 将 Tokenizer 的所有相关文件（vocab.json, merges.txt, tokenizer_config.json 等）保存到本地
        tokenizer.save_pretrained(SAVE_DIR)

        print(f"\n✅ Tokenizer 已成功下载并保存至真实路径:\n   {SAVE_DIR}")

        # 简单验证一下加载是否正常
        print("\n🔍 正在进行本地加载验证...")
        local_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)

        # 增加 padding token 设置（如果你的预训练代码需要 padding 的话）
        if local_tokenizer.pad_token is None:
            local_tokenizer.pad_token = local_tokenizer.eos_token
            print("   ⚠️ 已将 pad_token 自动设置为 eos_token。")

        test_text = "Testing the GPT-NeoX tokenizer for MoE architecture."
        tokens = local_tokenizer.encode(test_text)
        print(f"   输入文本: '{test_text}'")
        print(f"   Token IDs: {tokens}")
        print("   解码测试: " + local_tokenizer.decode(tokens))

        print("\n🎉 下载与验证全部完成！你可以在训练脚本中离线使用了。")

    except Exception as e:
        print(f"\n❌ 下载失败。错误信息:\n{e}")
        print("\n💡 诊断建议：")
        print("1. 如果仍然报 Network is unreachable，说明当前服务器(georgy)完全没有外网访问权限。")
        print(
            "2. 这种情况下，请在有网的电脑上运行此脚本，然后将生成的整个 gpt-neox-20b 文件夹通过 SCP/SFTP 传到服务器的对应目录。")


if __name__ == "__main__":
    download_and_save_tokenizer()