import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertConfig


# class MultitaskProtBertConfig(BertConfig):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.num_mechanism_labels = kwargs.get("num_mechanism_labels", 8)
#         self.num_antibiotic_labels = kwargs.get("num_antibiotic_labels", 48)
#         self.use_remove = kwargs.get("use_remove", True)


class ProteinDataset(Dataset):
    def __init__(self,
                 sequences,            # List[str]
                 is_arg,               # List[int]
                 mechanism_labels,     # List[List[float]]
                 antibiotic_labels,    # List[List[float]]
                 remove_labels,        # List[int]
                 tokenizer,
                 max_length=1024,
                 mask_ratio=0.05,       # 新增参数
                 training=True):        # 区分训练/验证阶段
        assert len(sequences) == len(is_arg) == len(mechanism_labels) == len(antibiotic_labels) == len(remove_labels)
        self.sequences = sequences
        self.is_arg = is_arg
        self.mechanism_labels = mechanism_labels
        self.antibiotic_labels = antibiotic_labels
        self.remove_labels = remove_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.training = training

        # 可替换氨基酸列表（只在Mask中使用）
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        label_arg = self.is_arg[idx]
        mech = self.mechanism_labels[idx]
        abc = self.antibiotic_labels[idx]
        rm = self.remove_labels[idx]

        # ==== 滑动窗口裁剪 ====
        if len(seq) > self.max_length:
            start = np.random.randint(0, len(seq) - self.max_length + 1)
            seq = seq[start:start + self.max_length]

        # ==== 随机 Mask（仅在训练阶段）====
        if self.training:
            seq = self.random_mask(seq, self.mask_ratio)

        # ==== Tokenization ====
        inputs = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids":        inputs["input_ids"].squeeze(0),
            "attention_mask":   inputs["attention_mask"].squeeze(0),
            "is_arg":           torch.tensor(label_arg, dtype=torch.long),
            "mechanism_labels": torch.tensor(mech, dtype=torch.float),
            "antibiotic_labels": torch.tensor(abc, dtype=torch.float),
            "remove_label":     torch.tensor(rm, dtype=torch.long)
        }


    def random_mask(self, seq, ratio):
        """随机将一定比例的氨基酸替换为 'X'，模拟不确定性"""
        seq = list(seq)
        for i in range(len(seq)):
            if seq[i] in self.amino_acids and np.random.rand() < ratio:
                seq[i] = 'X'
        return ''.join(seq)


def compute_remove_pos_weight(dataset):
    """
    计算remove任务的类别权重（只针对抗性样本）

    参数:
        dataset (ProteinDataset): 蛋白质数据集实例

    返回:
        torch.Tensor: remove任务的类别权重
    """
    # 提取所有样本的标签
    all_is_arg = []
    all_remove = []
    for i in range(len(dataset)):
        sample = dataset[i]
        all_is_arg.append(sample['is_arg'].item())
        all_remove.append(sample['remove_label'].item())

    # 筛选抗性样本的remove标签
    arg_remove_labels = [
        remove_label
        for i, remove_label in enumerate(all_remove)
        if all_is_arg[i] == 1
    ]

    # 验证数据分布
    total_arg_samples = len(arg_remove_labels)
    if total_arg_samples == 0:
        print("警告：数据集中没有抗性样本！使用默认权重1.0")
        return torch.tensor([1.0])

    arg_remove_pos = sum(arg_remove_labels)
    arg_remove_neg = total_arg_samples - arg_remove_pos

    print(f"Remove标签分布（仅抗性样本）: "
          f"正样本({arg_remove_pos}/{total_arg_samples}, {arg_remove_pos / total_arg_samples:.2%}), "
          f"负样本({arg_remove_neg}/{total_arg_samples}, {arg_remove_neg / total_arg_samples:.2%})")

    # 计算权重
    pos_weight = arg_remove_neg / arg_remove_pos
    pos_weight = min(pos_weight, 10.0)  # 限制最大权重值

    print(f"计算得到的权重: {pos_weight:.4f}")
    return torch.tensor([pos_weight], dtype=torch.float32)
def compute_pos_weight(label_matrix, clip_max=10):
    # label_matrix: numpy array (N, C)
    pos_counts = label_matrix.sum(axis=0)
    total = label_matrix.shape[0]

    # 避免除以0
    pos_counts[pos_counts == 0] = 1
    pos_weight = (total - pos_counts) / pos_counts

    # Clip 权重上限，防止 rare label 权重炸裂
    pos_weight = np.clip(pos_weight, a_min=1.0, a_max=clip_max)

    return torch.tensor(pos_weight, dtype=torch.float32)

def preprocess_data(csv_path, tokenizer_name, max_length=1024, training=True):
    df = pd.read_csv(csv_path)

    def parse_multilabel(x):
        x = x.strip("[]").split()
        return [float(i) for i in x]

    df['mechanism_encoded'] = df['rm_encoded'].apply(parse_multilabel)
    df['class_encoded'] = df['dc_encoded'].apply(parse_multilabel)

    sequences         = df["Sequence"].tolist()
    is_arg            = df["is_ARG"].tolist()
    mechanism_labels  = df["mechanism_encoded"].tolist()
    antibiotic_labels = df["class_encoded"].tolist()
    remove_labels     = df["remove_encoded"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return ProteinDataset(
        sequences,
        is_arg,
        mechanism_labels,
        antibiotic_labels,
        remove_labels,
        tokenizer,
        max_length=max_length,
        training=training,              # 👈 是否做随机mask增强
        mask_ratio=0.05                 # 可调
    )


if __name__ == "__main__":
    splits = ["train", "val", "test","my_test_encoded"]
    model_path = "/liymai24/hjh/codes/kkkk/ESM2_t30_150M_UR50D"  # 或你的本地模型路径

    for sp in splits:
        csv_path = f"data/{sp}.csv"
        print("Processing", csv_path)

        # 👇 判断当前是否为训练集
        is_training = (sp == "train")

        # 👇 将 training 参数传入
        ds = preprocess_data(csv_path, tokenizer_name=model_path, max_length=1024, training=is_training)

        out = f"processed_data/{sp}_dataset.pt"
        torch.save(ds, out)
        print("Saved", out)

        if sp == "train":
            print("Calculating positive weights for", sp)

            mechanism_labels = torch.stack([sample['mechanism_labels'] for sample in ds])
            antibiotic_labels = torch.stack([sample['antibiotic_labels'] for sample in ds])

            pos_weight_mech = compute_pos_weight(mechanism_labels)
            pos_weight_anti = compute_pos_weight(antibiotic_labels)
            pos_weight_remove = compute_remove_pos_weight(ds)

            print("pos_weight_mech", pos_weight_mech)
            print("pos_weight_anti", pos_weight_anti)
            print("pos_weight_remove", pos_weight_remove)

            torch.save(pos_weight_mech, 'processed_data/pos_weight_mech.pt')
            torch.save(pos_weight_anti, 'processed_data/pos_weight_anti.pt')
            torch.save(pos_weight_remove, 'processed_data/pos_weight_remove.pt')
            print("Saved positive weights")