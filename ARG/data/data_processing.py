import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

# === 清洗函数 & 标签替换同原版 ===
def clean_amino_acid_sequence(sequence):
    if not isinstance(sequence, str):
        sequence = str(sequence)
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    sequence = sequence.upper()
    sequence = re.sub(r'[^A-Z]', '', sequence)
    return ''.join(char for char in sequence if char in standard_aa)

def replace_rare(labels, rare_set):
    return sorted(set(["Other" if label in rare_set else label for label in labels]))

# === 读取数据 ===
pos_df = pd.read_csv("myarg.csv")
neg_df = pd.read_csv("final_negative_dataset.csv")

pos_df["Sequence"] = pos_df["Sequence"].apply(clean_amino_acid_sequence)
neg_df["Sequence"] = neg_df["Sequence"].apply(clean_amino_acid_sequence)
pos_df = pos_df[pos_df["Sequence"].str.len() > 0]
neg_df = neg_df[neg_df["Sequence"].str.len() > 0]

pos_df["is_ARG"] = 1
neg_df["is_ARG"] = 0
neg_df["Drug Class"] = ""
neg_df["Resistance Mechanism"] = ""
neg_df["Remove"] = 0

full_df = pd.concat([pos_df, neg_df], ignore_index=True)

# === multilabel 列解析与稀有标签替换 ===
full_df["dc_labels"] = full_df["Drug Class"].apply(lambda x: [i.strip() for i in x.split(";")] if isinstance(x, str) and x.strip() else [])
full_df["rm_labels"] = full_df["Resistance Mechanism"].apply(lambda x: [i.strip() for i in x.split(";")] if isinstance(x, str) and x.strip() else [])

dc_counter = Counter(label for labels in full_df.loc[full_df["is_ARG"] == 1, "dc_labels"] for label in labels)
rm_counter = Counter(label for labels in full_df.loc[full_df["is_ARG"] == 1, "rm_labels"] for label in labels)
dc_threshold, rm_threshold = 50, 50
rare_dc = {lab for lab, cnt in dc_counter.items() if cnt < dc_threshold}
rare_rm = {lab for lab, cnt in rm_counter.items() if cnt < rm_threshold}

full_df["dc_labels"] = full_df["dc_labels"].apply(lambda x: replace_rare(x, rare_dc))
full_df["rm_labels"] = full_df["rm_labels"].apply(lambda x: replace_rare(x, rare_rm))
full_df["Drug Class"] = full_df["dc_labels"].apply(lambda x: ";".join(sorted(set(x))))
full_df["Resistance Mechanism"] = full_df["rm_labels"].apply(lambda x: ";".join(sorted(set(x))))

# === 数据集拆分 ===
train_val_df, test_df = train_test_split(full_df, test_size=0.15,
                                         stratify=full_df["is_ARG"], random_state=45)
train_df, val_df = train_test_split(train_val_df, test_size=0.176,
                                    stratify=train_val_df["is_ARG"], random_state=45)

def get_covered_labels(df, key):
    arr = np.stack(df[key].values)
    return set(np.where(arr.sum(axis=0) > 0)[0])

def move_samples_to_cover(train_df, val_df, test_df, key):
    total = set(range(len(train_df[key].iloc[0])))
    train_cov = get_covered_labels(train_df, key)
    for name, df_other in [("val", val_df), ("test", test_df)]:
        to_move = []
        for idx, row in df_other.iterrows():
            labs = set(np.where(row[key] > 0)[0])
            if not labs.issubset(train_cov):
                train_cov.update(labs)
                to_move.append(idx)
                if train_cov == total:
                    break
        if to_move:
            df_move = df_other.loc[to_move]
            train_df = pd.concat([train_df, df_move], ignore_index=True)
            if name == "val":
                val_df = df_other.drop(to_move)
            else:
                test_df = df_other.drop(to_move)
    return train_df, val_df, test_df

# 先暂时放空编码器列供后续填充
for df in [train_df, val_df, test_df]:
    df["dc_encoded"] = None
    df["rm_encoded"] = None

# === one-hot 模型上的标签编码 ===
# 先创建两个 MultiLabelBinarizer，后面在 train 上 fit
dc_encoder = MultiLabelBinarizer()
rm_encoder = MultiLabelBinarizer()

# 临时把列表形式保存为原始列给 fit 用
train_df_refit = train_df.copy()
train_df_refit["dc_list"] = train_df_refit["dc_labels"]
train_df_refit["rm_list"] = train_df_refit["rm_labels"]

dc_encoder.fit(train_df_refit.loc[train_df_refit["is_ARG"] == 1, "dc_list"])
rm_encoder.fit(train_df_refit.loc[train_df_refit["is_ARG"] == 1, "rm_list"])

# 然后对所有 splits 做编码并写入新列
for df in [train_df, val_df, test_df]:
    df["dc_encoded"] = df["dc_labels"].apply(
        lambda x: dc_encoder.transform([x])[0] if df.loc[df.index[0], "is_ARG"] == 1 or True else np.zeros(len(dc_encoder.classes_))
    )
    df["rm_encoded"] = df["rm_labels"].apply(
        lambda x: rm_encoder.transform([x])[0] if df.loc[df.index[0], "is_ARG"] == 1 or True else np.zeros(len(rm_encoder.classes_))
    )
    df["remove_encoded"] = df["Remove"].fillna(0).astype(int)
    df.loc[df["is_ARG"] == 0, "remove_encoded"] = 0

# 覆盖训练集确保规则
train_df, val_df, test_df = move_samples_to_cover(train_df, val_df, test_df, "dc_encoded")
train_df, val_df, test_df = move_samples_to_cover(train_df, val_df, test_df, "rm_encoded")

# === 保存 splits 和标签脚本 ===
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    df.to_csv(f"{name}.csv", index=False)

with open("labels.py", "w") as f:
    f.write("mechanism_labels = [\n")
    for lab in rm_encoder.classes_:
        f.write(f'    "{lab}",\n')
    f.write("]\n\nantibiotic_labels = [\n")
    for lab in dc_encoder.classes_:
        f.write(f'    "{lab}",\n')
    f.write("]\n\nremove_labels = [\"0\", \"1\"]\n")

print(f"训练集抗生素标签覆盖: {len(get_covered_labels(train_df, 'dc_encoded'))} / {len(dc_encoder.classes_)}")
print(f"训练集抗性机制标签覆盖: {len(get_covered_labels(train_df, 'rm_encoded'))} / {len(rm_encoder.classes_)}")


# === 添加统计功能 ===
def print_label_distribution(df, df_name):
    """打印数据集标签分布统计"""
    print(f"\n=== {df_name} 数据集分布 ===")
    print(f"总样本数: {len(df)}")
    print(f"正样本 (ARG): {df['is_ARG'].sum()}, 负样本: {len(df) - df['is_ARG'].sum()}")

    # 统计每个抗生素类别的分布
    if 'dc_encoded' in df.columns:
        dc_counts = np.sum(np.vstack(df['dc_encoded'].values), axis=0)
        print("\n抗生素类别分布:")
        for label, count in zip(dc_encoder.classes_, dc_counts):
            print(f"  {label}: {int(count)}个样本")

    # 统计每个抗性机制的分布
    if 'rm_encoded' in df.columns:
        rm_counts = np.sum(np.vstack(df['rm_encoded'].values), axis=0)
        print("\n抗性机制分布:")
        for label, count in zip(rm_encoder.classes_, rm_counts):
            print(f"  {label}: {int(count)}个样本")

    # 统计移除标签的分布
    if 'Remove' in df.columns:
        remove_counts = df['Remove'].value_counts().sort_index()
        print("\n移除标签分布:")
        for value, count in remove_counts.items():
            print(f"  {value}: {count}个样本")

    print("=" * 50)


# 打印各数据集分布
print_label_distribution(train_df, "训练集")
print_label_distribution(val_df, "验证集")
print_label_distribution(test_df, "测试集")