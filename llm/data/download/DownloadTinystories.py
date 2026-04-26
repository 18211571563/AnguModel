import os
from datasets import load_dataset

# ==========================================
# ⚙️ 配置区域
# ==========================================
# 1. 指定你的本地保存路径（建议放在 SSD 或挂载的独立数据盘下）
SAVE_DIR = "/home/georgy/model/data/train_data/TinyStories"

# 2. 针对国内网络环境，配置 HuggingFace 镜像源加速下载
# 如果你在公司内网有代理，可以注释掉这行；如果没有，强烈建议保留
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_and_save_dataset():
    print("🚀 正在连接 HuggingFace 镜像源请求 TinyStories...")

    try:
        # load_dataset 会自动下载并缓存数据
        # num_proc=8 开启多进程下载和解析，大幅提升下载速度
        dataset = load_dataset("roneneldan/TinyStories", num_proc=8)

        print("\n✅ 数据集拉取成功！数据结构如下：")
        print(dataset)

        print(f"\n💾 正在将数据集以 Arrow 格式持久化保存至: {SAVE_DIR}")
        print("   (这可能需要几分钟，请耐心等待...)")

        # 将数据集保存到指定目录
        dataset.save_to_disk(SAVE_DIR)

        print("\n🎉 下载与本地化保存全部完成！")

    except Exception as e:
        print(f"\n❌ 下载过程中出现错误: {e}")
        print("提示：如果遇到证书错误，请检查网络代理；如果遇到 404，请确认 hf-mirror 镜像源状态。")


if __name__ == "__main__":
    # 确保保存的父目录存在
    os.makedirs(os.path.dirname(SAVE_DIR), exist_ok=True)
    download_and_save_dataset()