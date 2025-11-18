import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# === 加载已有编码器（如果之前保存了）===
# dc_encoder = joblib.load("dc_encoder.pkl")
# rm_encoder = joblib.load("rm_encoder.pkl")

# 或者从已有的 train.csv 构建编码器
train_df = pd.read_csv("data/train.csv")

def parse_label(x):
    return [i.strip() for i in x.split(";")] if isinstance(x, str) and x.strip() else []

train_df["dc_labels"] = train_df["Drug Class"].apply(parse_label)
train_df["rm_labels"] = train_df["Resistance Mechanism"].apply(parse_label)

dc_encoder = MultiLabelBinarizer()
rm_encoder = MultiLabelBinarizer()
dc_encoder.fit(train_df.loc[train_df["is_ARG"] == 1, "dc_labels"])
rm_encoder.fit(train_df.loc[train_df["is_ARG"] == 1, "rm_labels"])

# === 读取你的新测试文件 ===
test_df = pd.read_csv("data/predict_data.csv")

# 清洗序列（可选）
def clean_amino_acid_sequence(sequence):
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    sequence = str(sequence).upper()
    sequence = re.sub(r'[^A-Z]', '', sequence)
    return ''.join([aa for aa in sequence if aa in standard_aa])

test_df["Sequence"] = test_df["Sequence"].apply(clean_amino_acid_sequence)

# 标签列解析
test_df["dc_labels"] = test_df["Drug Class"].apply(parse_label)
test_df["rm_labels"] = test_df["Resistance Mechanism"].apply(parse_label)

# 标签编码（若为非 ARG，置 0 向量）
num_dc = len(dc_encoder.classes_)
num_rm = len(rm_encoder.classes_)

test_df["dc_encoded"] = test_df.apply(
    lambda row: dc_encoder.transform([row["dc_labels"]])[0] if row["is_ARG"] == 1 else np.zeros(num_dc),
    axis=1
)
test_df["rm_encoded"] = test_df.apply(
    lambda row: rm_encoder.transform([row["rm_labels"]])[0] if row["is_ARG"] == 1 else np.zeros(num_rm),
    axis=1
)

# Remove 编码
test_df["remove_encoded"] = test_df["Remove"].fillna(0).astype(int)
test_df.loc[test_df["is_ARG"] == 0, "remove_encoded"] = 0

# 可选：保存为新文件
test_df.to_csv("my_test_encoded.csv", index=False)

print("测试集编码完成 ✅")
