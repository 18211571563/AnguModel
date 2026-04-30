from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk


class TinyStoriesDataSetProcess:
    def __init__(self):
        super().__init__()

    @staticmethod
    def process(tokenizer: PreTrainedTokenizerFast,local_data_path, max_length, batch_size, load_data_size = None) -> DataLoader:
        # 1. 从本地或缓存加载数据集 (假设你已经下载过了，它会自动走本地缓存)
        dataset = load_from_disk(local_data_path)

        # 根据你本地数据的实际情况决定是否需要这一句
        # 如果本地保存的数据集包含 train/test split，需要显式指定：
        dataset = dataset["train"]

        # ⚠️ 为了测试跑通，建议先取一个子集，比如前 100,000 条。
        # 等代码完全跑通且不 OOM 时，再把下面这行注释掉，跑全量。
        if load_data_size is not None:
            dataset = dataset.select(range(load_data_size))

        print(f"✅ 成功加载了 {len(dataset)} 条故事。")

        # 2. 定义数据预处理函数 (Tokenization)
        def tokenize_function(examples):
            # 💡 核心：给每个故事强制拼上 eos_token！
            texts = [text + tokenizer.eos_token for text in examples["text"]]

            # 批量进行 Tokenize，并统一截断和填充到最大长度
            return tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )


        print("⏳ 正在对数据集进行 Tokenize 编码 (多进程加速)...")
        # 3. 使用 map 进行多进程高效处理
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,  # 开启批量处理
            num_proc=8,  # 你的 8 进程加速
            remove_columns=["text"]  # 处理完后，丢弃原始文本列，节省内存
        )

        # 4. 将数据集格式转换为 PyTorch Tensor 格式
        tokenized_dataset.set_format("torch", columns=["input_ids"])

        # 5. 构建 PyTorch DataLoader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True  # 丢弃最后不够一个 Batch 的数据，保持张量形状绝对一致
        )
        print("✅ 数据加载与 DataLoader 构建完成！")

        return dataloader
