import json
import os


class Config:
    def __init__(self, **kwargs):
        # 1. 设置默认值（兜底机制，防止 JSON 里漏写）
        self.model_type = "my_llama_moe"
        self.vocab_size = 21128
        self.dim = 512
        self.hidden_dim = 1376
        self.layer_num = 4
        self.head_num = 8
        self.max_seq_len = 1024
        self.pad_token_id = 0
        self.eos_token_id = 2

        # MoE 专属参数
        self.expert_num = 8
        self.shared_num = 1
        self.top_k = 2

        # 2. 用传入的 kwargs 覆盖默认值
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, json_path: str):
        """
        前沿规范：提供从文件加载的类方法 (模仿 HuggingFace)
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"配置文件 {json_path} 不存在！")

        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # 实例化类并返回
        return cls(**config_dict)

    def save_pretrained(self, save_dir: str):
        """
        前沿规范：提供将当前内存中的配置保存回硬盘的方法
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "config.json")
        with open(save_path, "w", encoding="utf-8") as f:
            # vars(self) 将对象的属性转换为字典
            json.dump(vars(self), f, indent=4, ensure_ascii=False)