import csv


def csv_to_fasta(csv_file, fasta_file):
    """
    将包含ID和Sequence列的CSV文件转换为FASTA格式

    参数:
    csv_file (str): 输入的CSV文件路径
    fasta_file (str): 输出的FASTA文件路径
    """
    with open(csv_file, 'r') as csv_in, open(fasta_file, 'w') as fasta_out:
        reader = csv.DictReader(csv_in)

        # 检查必要的列是否存在
        if 'ID' not in reader.fieldnames or 'Sequence' not in reader.fieldnames:
            raise ValueError("CSV文件必须包含'ID'和'Sequence'列")

        for row in reader:
            # 写入FASTA头部（以>开头）
            fasta_out.write(f'>{row["ID"]}\n')

            # 按每行80字符分割序列（标准FASTA格式）
            sequence = row["Sequence"].strip()
            for i in range(0, len(sequence), 80):
                fasta_out.write(sequence[i:i + 80] + '\n')


# 示例使用
if __name__ == "__main__":
    input_csv = "data.csv"  # 替换为你的输入CSV文件路径
    output_fasta = "data.fasta"  # 替换为你的输出FASTA文件路径

    csv_to_fasta(input_csv, output_fasta)
    print(f"转换完成！FASTA文件已保存至: {output_fasta}")