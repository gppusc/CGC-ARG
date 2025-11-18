import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertConfig


class MultitaskProtBertConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_mechanism_labels = kwargs.get("num_mechanism_labels", 8)
        self.num_antibiotic_labels = kwargs.get("num_antibiotic_labels", 48)
        self.use_remove = kwargs.get("use_remove", True)


class ProteinDataset(Dataset):
    def __init__(self,
                 sequences,            # List[str]
                 is_arg,               # List[int]
                 mechanism_labels,     # List[List[float]]
                 antibiotic_labels,    # List[List[float]]
                 remove_labels,        # List[int]
                 tokenizer,
                 max_length=1024):
        assert len(sequences) == len(is_arg) == len(mechanism_labels) == len(antibiotic_labels) == len(remove_labels)
        self.sequences = sequences
        self.is_arg = is_arg
        self.mechanism_labels = mechanism_labels
        self.antibiotic_labels = antibiotic_labels
        self.remove_labels = remove_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        label_arg = self.is_arg[idx]
        mech = self.mechanism_labels[idx]
        abc = self.antibiotic_labels[idx]
        rm = self.remove_labels[idx]

        # 直接使用 amino acid 序列，无需 k-mer
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
def compute_pos_weight(label_matrix, clip_max=15.0):
    # label_matrix: numpy array (N, C)
    pos_counts = label_matrix.sum(axis=0)
    total = label_matrix.shape[0]

    # 避免除以0
    pos_counts[pos_counts == 0] = 1
    pos_weight = (total - pos_counts) / pos_counts

    # Clip 权重上限，防止 rare label 权重炸裂
    pos_weight = np.clip(pos_weight, a_min=1.0, a_max=clip_max)

    return torch.tensor(pos_weight, dtype=torch.float32)

def preprocess_data(csv_path, tokenizer_name, max_length=1024):
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
        max_length=max_length
    )


if __name__ == "__main__":
    splits = ["train", "val", "test"]
    model_path = "ESM2_T6_8M_UR50D"  # 或你的本地模型路径

    for sp in splits:
        csv_path = f"data/{sp}.csv"
        print("Processing", csv_path)
        ds = preprocess_data(csv_path, tokenizer_name=model_path, max_length=1024)
        out = f"processed_data/{sp}_dataset.pt"
        torch.save(ds, out)
        print("Saved", out)
        if sp == "train":
            print("Calculating positive weights for", sp)

            # 假设数据集返回字典包含 'mechanism' 和 'antibiotic' 的标签
            # 提取所有标签（假设标签存储为张量）
            mechanism_labels = torch.stack([sample['mechanism_labels'] for sample in ds])
            antibiotic_labels = torch.stack([sample['antibiotic_labels'] for sample in ds])

            # 计算正类权重
            pos_weight_mech = compute_pos_weight(mechanism_labels)
            pos_weight_anti = compute_pos_weight(antibiotic_labels)


            print("pos_weight_mech", pos_weight_mech)
            print("pos_weight_anti", pos_weight_anti)
            torch.save(pos_weight_mech,
                       'processed_data/pos_weight_mech.pt')
            torch.save(pos_weight_anti,
                       'processed_data/pos_weight_anti.pt')


            print("Saved positive weights ")
