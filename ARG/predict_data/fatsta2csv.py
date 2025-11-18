import csv


def fasta_to_csv(fasta_file, csv_file):
    """
    将FASTA文件转换为包含ID和Sequence列的CSV文件

    参数:
    fasta_file (str): 输入的FASTA文件路径
    csv_file (str): 输出的CSV文件路径
    """
    with open(fasta_file, 'r') as fasta_in, open(csv_file, 'w', newline='') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(['ID', 'Sequence'])  # 写入CSV表头

        current_id = None
        current_seq = []

        for line in fasta_in:
            line = line.strip()  # 移除首尾空白字符

            if line.startswith('>'):  # 头部行
                # 如果已有序列在缓存中，先写入前一个序列
                if current_id is not None:
                    writer.writerow([current_id, ''.join(current_seq)])
                    current_seq = []  # 重置序列缓存

                # 获取新ID（移除开头的>符号和可能的注释）
                header = line[1:].strip()
                # 提取第一个空格前的部分作为ID（常见FASTA格式）
                current_id = header.split()[0] if ' ' in header else header
            else:
                # 序列行（跳过空行）
                if line:
                    current_seq.append(line)

        # 写入最后一个序列
        if current_id is not None:
            writer.writerow([current_id, ''.join(current_seq)])


# 示例使用
if __name__ == "__main__":
    input_fasta = "data_aa.fasta"  # 替换为你的输入FASTA文件路径
    output_csv = "dtata_aa.csv"  # 替换为你的输出CSV文件路径

    fasta_to_csv(input_fasta, output_csv)
    print(f"转换完成！CSV文件已保存至: {output_csv}")