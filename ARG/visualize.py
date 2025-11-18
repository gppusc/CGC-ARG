"""
抗生素类别特征可视化分析工具（修复版本）
使用示例：python visualize.py --model_path models/best_model --target_classes fluoroquinolone,quinolone
"""

import argparse
import sys

import numpy as np
# 修复Matplotlib后端错误
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
# 其余导入保持不变
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from Dataset import ProteinDataset  # 假设有数据集类
sys.path.append('data')
from labels import mechanism_labels, antibiotic_labels
def get_features(model, dataloader, target_classes, antibiotic_classes):
    """获取模型中间层的特征表示"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 收集特征和标签
    features = []
    class_labels = []

    # 获取目标类别的索引
    target_indices = [antibiotic_classes.index(cls) for cls in target_classes]

    # 使用钩子捕获中间层输出
    activation = {}

    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
        return hook

    # 注册钩子到倒数第二层
    try:
        # 尝试获取ESM模型的编码器层
        hook = model.esm.encoder.layer[-2].register_forward_hook(get_activation('features'))
    except AttributeError:
        # 如果ESM结构不同，尝试直接访问最后一层
        try:
            hook = model.esm.register_forward_hook(get_activation('features'))
        except Exception as e:
            print(f"错误: 无法注册钩子 - {str(e)}")
            return np.array([]), np.array([])

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            # 前向传播
            model(**inputs)

            # 获取捕获的特征
            if 'features' in activation:
                # 获取[CLS] token特征 (批大小, 隐藏层大小)
                cls_features = activation['features'][:, 0, :]
                features.append(cls_features.cpu().numpy())

            # 创建标签向量
            batch_labels = np.zeros(len(batch["antibiotic_labels"]))
            for i, class_idx in enumerate(target_indices):
                class_mask = batch["antibiotic_labels"][:, class_idx] == 1
                batch_labels[class_mask] = i + 1

            class_labels.append(batch_labels)

    # 移除钩子
    hook.remove()

    if len(features) == 0:
        return np.array([]), np.array([])

    features = np.vstack(features)
    class_labels = np.concatenate(class_labels)

    return features, class_labels

def visualize_features(model, dataloader, target_classes, antibiotic_classes, output_dir="analysis"):
    """执行PCA/t-SNE可视化分析"""
    # 获取特征和标签
    features, labels = get_features(model, dataloader, target_classes, antibiotic_classes)

    if len(features) == 0:
        print("错误：未能获取特征。请检查模型结构。")
        return

    # 筛选目标类别样本
    target_mask = labels > 0
    target_features = features[target_mask]
    target_labels = labels[target_mask]

    if len(target_features) == 0:
        print(f"警告：未找到目标类别 {target_classes} 的样本")
        return

    # 确保输出目录存在
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 执行PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(target_features)
    plot_results(pca_result, target_labels, target_classes, f"{output_dir}/pca_plot.png", "PCA")

    # 执行t-SNE
    perplexity = min(30, len(target_features)-1) if len(target_features) > 1 else 1
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000)
    tsne_result = tsne.fit_transform(target_features)
    plot_results(tsne_result, target_labels, target_classes, f"{output_dir}/tsne_plot.png", "t-SNE")

def plot_results(coords, labels, class_names, save_path, method_name):
    """可视化结果并保存"""
    plt.figure(figsize=(12, 10))

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, class_name in enumerate(class_names):
        class_mask = labels == (i + 1)
        if np.sum(class_mask) > 0:  # 确保有样本
            plt.scatter(
                coords[class_mask, 0], coords[class_mask, 1],
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                alpha=0.7,
                s=80,
                label=f"{class_name} (n={np.sum(class_mask)})"
            )

    plt.title(f'{method_name} Visualization of Antibiotic Classes', fontsize=14)
    plt.xlabel(f'{method_name} Component 1', fontsize=12)
    plt.ylabel(f'{method_name} Component 2', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # 添加分离度分析
    if len(class_names) > 1:
        separation_score = calculate_separation(coords, labels)
        plt.figtext(0.5, 0.01, f"Class Separation Score: {separation_score:.2f}",
                    ha="center", fontsize=11, style='italic')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形释放内存
    print(f"Saved {method_name} plot to {save_path}")

def calculate_separation(coords, labels):
    """计算类间分离度分数"""
    from sklearn.neighbors import NearestNeighbors
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0  # 只有一个类别时无法计算分离度

    intra_dists, inter_dists = [], []

    for label in unique_labels:
        class_points = coords[labels == label]

        # 类内距离"""
        # 抗生素类别特征可视化分析工具
        # 使用示例：python visualize.py --model_path models/best_model --target_classes fluoroquinolone,quinolone
        # """
        #
        # import argparse
        # import numpy as np
        # # 修复Matplotlib后端错误
        # import matplotlib
        # matplotlib.use('Agg')  # 使用非交互式后端
        # import matplotlib.pyplot as plt
        # from sklearn.decomposition import PCA
        # from sklearn.manifold import TSNE
        # import torch
        # from torch.utils.data import DataLoader
        # from Dataset import ProteinDataset  # 假设有数据集类
        #
        # def get_features(model_1, dataloader, target_classes, antibiotic_classes):
        #     """获取模型中间层的特征表示"""
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     model_1.to(device)
        #     model_1.eval()
        #
        #     # 收集特征和标签
        #     features = []
        #     class_labels = []
        #
        #     # 获取目标类别的索引
        #     target_indices = [antibiotic_classes.index(cls) for cls in target_classes]
        #
        #     # 使用钩子捕获抗生素分类任务的特征（f_abc）
        #     activation = {}
        #
        #     def get_activation(name):
        #         def hook(model_1, input, output):
        #             # 直接捕获输出张量
        #             activation[name] = output.detach()
        #         return hook
        #
        #     # 注册钩子到抗生素任务特征层
        #     try:
        #         # 尝试获取抗生素分类任务的特征（f_abc）
        #         hook = model_1.task_norm_abc.register_forward_hook(get_activation('antibiotic_features'))
        #     except AttributeError:
        #         try:
        #             # 如果找不到，尝试获取抗生素分类头之前的特征
        #             hook = model_1.antibiotic_head.register_forward_hook(get_activation('antibiotic_features'))
        #         except Exception as e:
        #             print(f"错误: 无法注册钩子 - {str(e)}")
        #             return np.array([]), np.array([])
        #
        #     with torch.no_grad():
        #         for batch_idx, batch in enumerate(dataloader):
        #             # 每次迭代前重置activation字典
        #             activation = {}
        #
        #             # 准备输入
        #             inputs = {
        #                 "input_ids": batch["input_ids"].to(device),
        #                 "attention_mask": batch["attention_mask"].to(device),
        #             }
        #
        #             # 添加标签（即使不用于计算损失，模型可能需要它们）
        #             labels = {
        #                 "resistance_labels": torch.zeros(len(batch["input_ids"])).to(device),
        #                 "mechanism_labels": torch.zeros((len(batch["input_ids"]), model_1.num_mechanism_labels)).to(device),
        #                 "antibiotic_labels": batch["antibiotic_labels"].to(device),
        #                 "remove_labels": torch.zeros(len(batch["input_ids"])).to(device),
        #             }
        #
        #             # 前向传播
        #             model_1(**inputs, **labels)
        #
        #             # 获取捕获的特征
        #             if 'antibiotic_features' in activation:
        #                 # 抗生素任务特征 (batch_size, feature_dim)
        #                 task_features = activation['antibiotic_features']
        #                 features.append(task_features.cpu().numpy())
        #             else:
        #                 print(f"警告: 批次 {batch_idx} 未捕获到特征")
        #                 continue
        #
        #             # 创建标签向量
        #             batch_labels = np.zeros(len(batch["antibiotic_labels"]))
        #             for i, class_idx in enumerate(target_indices):
        #                 class_mask = batch["antibiotic_labels"][:, class_idx] == 1
        #                 batch_labels[class_mask] = i + 1
        #
        #             class_labels.append(batch_labels)
        #
        #     # 移除钩子
        #     hook.remove()
        #
        #     if len(features) == 0:
        #         return np.array([]), np.array([])
        #
        #     features = np.vstack(features)
        #     class_labels = np.concatenate(class_labels)
        #
        #     return features, class_labels
        #
        # def visualize_features(model_1, dataloader, target_classes, antibiotic_classes, output_dir="analysis"):
        #     """执行PCA/t-SNE可视化分析"""
        #     # 获取特征和标签
        #     features, labels = get_features(model_1, dataloader, target_classes, antibiotic_classes)
        #
        #     if len(features) == 0:
        #         print("错误：未能获取特征。请检查模型结构。")
        #         return
        #
        #     # 筛选目标类别样本
        #     target_mask = labels > 0
        #     target_features = features[target_mask]
        #     target_labels = labels[target_mask]
        #
        #     if len(target_features) == 0:
        #         print(f"警告：未找到目标类别 {target_classes} 的样本")
        #         return
        #
        #     # 确保输出目录存在
        #     import os
        #     os.makedirs(output_dir, exist_ok=True)
        #
        #     # 执行PCA
        #     pca = PCA(n_components=2)
        #     pca_result = pca.fit_transform(target_features)
        #     plot_results(pca_result, target_labels, target_classes, f"{output_dir}/pca_plot.png", "PCA")
        #
        #     # 执行t-SNE
        #     perplexity = min(30, len(target_features)-1) if len(target_features) > 1 else 1
        #     tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
        #     tsne_result = tsne.fit_transform(target_features)
        #     plot_results(tsne_result, target_labels, target_classes, f"{output_dir}/tsne_plot.png", "t-SNE")
        #
        # def plot_results(coords, labels, class_names, save_path, method_name):
        #     """可视化结果并保存"""
        #     plt.figure(figsize=(12, 10))
        #
        #     colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        #     markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        #
        #     for i, class_name in enumerate(class_names):
        #         class_mask = labels == (i + 1)
        #         if np.sum(class_mask) > 0:  # 确保有样本
        #             plt.scatter(
        #                 coords[class_mask, 0], coords[class_mask, 1],
        #                 c=colors[i % len(colors)],
        #                 marker=markers[i % len(markers)],
        #                 alpha=0.7,
        #                 s=80,
        #                 label=f"{class_name} (n={np.sum(class_mask)})"
        #             )
        #
        #     plt.title(f'{method_name} Visualization of Antibiotic Classes', fontsize=14)
        #     plt.xlabel(f'{method_name} Component 1', fontsize=12)
        #     plt.ylabel(f'{method_name} Component 2', fontsize=12)
        #     plt.legend(fontsize=10, loc='best')
        #     plt.grid(alpha=0.2)
        #     plt.tight_layout()
        #
        #     # 添加分离度分析
        #     if len(class_names) > 1:
        #         separation_score = calculate_separation(coords, labels)
        #         plt.figtext(0.5, 0.01, f"Class Separation Score: {separation_score:.2f}",
        #                     ha="center", fontsize=11, style='italic')
        #
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     plt.close()  # 关闭图形释放内存
        #     print(f"Saved {method_name} plot to {save_path}")
        #
        # def calculate_separation(coords, labels):
        #     """计算类间分离度分数"""
        #     from sklearn.neighbors import NearestNeighbors
        #     unique_labels = np.unique(labels)
        #
        #     if len(unique_labels) < 2:
        #         return 0.0  # 只有一个类别时无法计算分离度
        #
        #     intra_dists, inter_dists = [], []
        #
        #     for label in unique_labels:
        #         class_points = coords[labels == label]
        #
        #         # 类内距离
        #         if len(class_points) > 1:
        #             nn = NearestNeighbors(n_neighbors=2).fit(class_points)
        #             distances, _ = nn.kneighbors(class_points)
        #             intra_dists.append(np.mean(distances[:, 1]))
        #         else:
        #             intra_dists.append(0.0)
        #
        #         # 类间距离
        #         other_points = coords[labels != label]
        #         if len(other_points) > 0:
        #             nn = NearestNeighbors(n_neighbors=1).fit(other_points)
        #             distances, _ = nn.kneighbors(class_points)
        #             inter_dists.append(np.mean(distances))
        #         else:
        #             inter_dists.append(0.0)
        #
        #     intra_mean = np.mean([d for d in intra_dists if d > 0])
        #     inter_mean = np.mean([d for d in inter_dists if d > 0])
        #
        #     if intra_mean == 0:  # 避免除以零
        #         return inter_mean if inter_mean > 0 else 0.0
        #
        #     return inter_mean / intra_mean
        #
        # if __name__ == "__main__":
        #     parser = argparse.ArgumentParser()
        #     parser.add_argument("--model_path", type=str, default="/liymai24/hjh/codes/kkkk/ARG_Cleaned/outputs/AutoCNN_NewMoE_ASL_outputs_4/epoch_models/epoch_11")
        #     parser.add_argument("--target_classes", type=str, default="fluoroquinolone,quinolone",
        #                         help="Comma-separated class names (e.g. 'fluoroquinolone,quinolone')")
        #     parser.add_argument("--dataset_path", type=str, default="processed_data/test_dataset.pt")
        #     parser.add_argument("--output_dir", type=str, default="analysis")
        #     args = parser.parse_args()
        #
        #     # 加载模型和数据
        #     from AutoCNN_NewMoE_ASL import GCM_MultiLabelModel
        #     model_1 = GCM_MultiLabelModel.from_pretrained(args.model_path)
        #     test_dataset = torch.load(args.dataset_path)
        #     test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        #
        #     # 获取类别索引
        #     try:
        #         ANTIBIOTIC_CLASSES = test_dataset.antibiotic_classes
        #     except AttributeError:
        #         # 创建默认类别列表
        #         ANTIBIOTIC_CLASSES = [
        #             "aminoglycoside", "bacitracin", "beta_lactam", "chloramphenicol",
        #             "fluoroquinolone", "fosfomycin", "glycopeptide", "macrolide",
        #             "macrolide_lincosamide_streptogramin", "multidrug", "peptide",
        #             "phenicol", "polymyxin", "quinolone", "rifampin", "sulfonamide",
        #             "tetracycline", "trimethoprim"
        #         ]
        #
        #     # 执行可视化
        #     target_classes = [c.strip() for c in args.target_classes.split(",")]
        #     print(f"开始可视化分析目标类别: {target_classes}")
        #     visualize_features(
        #         model_1,
        #         test_loader,
        #         target_classes,
        #         ANTIBIOTIC_CLASSES,  # 添加抗生素类别列表
        #         args.output_dir
        #     )
        #     print("可视化分析完成")
        if len(class_points) > 1:
            nn = NearestNeighbors(n_neighbors=2).fit(class_points)
            distances, _ = nn.kneighbors(class_points)
            intra_dists.append(np.mean(distances[:, 1]))
        else:
            intra_dists.append(0.0)

        # 类间距离
        other_points = coords[labels != label]
        if len(other_points) > 0:
            nn = NearestNeighbors(n_neighbors=1).fit(other_points)
            distances, _ = nn.kneighbors(class_points)
            inter_dists.append(np.mean(distances))
        else:
            inter_dists.append(0.0)

    intra_mean = np.mean([d for d in intra_dists if d > 0])
    inter_mean = np.mean([d for d in inter_dists if d > 0])

    if intra_mean == 0:  # 避免除以零
        return inter_mean if inter_mean > 0 else 0.0

    return inter_mean / intra_mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/liymai24/hjh/codes/kkkk/ARG_Cleaned/outputs/AutoCNN_NewMoE_ASL_outputs_4/epoch_models/epoch_11")
    parser.add_argument("--target_classes", type=str, default="fluoroquinolone,quinolone",
                        help="Comma-separated class names (e.g. 'fluoroquinolone,quinolone')")
    parser.add_argument("--dataset_path", type=str, default="processed_data/train_dataset.pt")
    parser.add_argument("--output_dir", type=str, default="analysis")
    args = parser.parse_args()

    # 加载模型和数据
    from AutoCNN_NewMoE_ASL import GCM_MultiLabelModel
    model = GCM_MultiLabelModel.from_pretrained(args.model_path)
    test_dataset = torch.load(args.dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 获取类别索引
    try:
        ANTIBIOTIC_CLASSES = test_dataset.antibiotic_classes
    except AttributeError:
        # 创建默认类别列表
        ANTIBIOTIC_CLASSES = antibiotic_labels

    # 执行可视化
    target_classes = [c.strip() for c in args.target_classes.split(",")]
    visualize_features(
        model,
        test_loader,
        target_classes,
        ANTIBIOTIC_CLASSES,  # 添加抗生素类别列表
        args.output_dir
    )