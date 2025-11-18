import csv
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig
import yaml
from tqdm import tqdm
from AutoCNN_NewMoE_ASL import GCM_MultiLabelModel
from utils_1 import compute_metrics
from Dataset import ProteinDataset
import os


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(state, checkpoint_dir="outputs/AutoCNN_NewMoE_ASL_outputs_1/checkpoints", filename="checkpoint.pth.tar"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, filename))


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    if os.path.isfile(checkpoint_path):
        print(f"=> 加载检查点 '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        return checkpoint['epoch'], checkpoint['best_val_loss'], checkpoint['best_rare_f1'], checkpoint[
            'early_stop_counter']
    else:
        print(f"=> 未找到检查点 '{checkpoint_path}'")
        return 0, float('inf'), -1.0, 0


def calculate_class_weights(dataset, num_classes, task_type="mechanism"):
    counts = torch.zeros(num_classes)
    for sample in dataset:
        if sample["is_arg"] == 1:
            if task_type == "mechanism":
                counts += sample["mechanism_labels"]
            elif task_type == "antibiotic":
                counts += sample["antibiotic_labels"]
    counts = torch.clamp(counts, min=1)
    weights = 1.0 / torch.sqrt(counts + 1e-5)
    weights = weights / weights.sum() * num_classes
    if task_type == "mechanism":
        return torch.cat([torch.tensor([1.0]), weights])
    elif task_type == "antibiotic":
        return torch.cat([torch.tensor([1.0]), weights])
    return weights


def train():
    torch.cuda.empty_cache()
    config = load_config("configASL.yaml")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    csv_file = "validation_metrics.csv"

    # 检查点配置
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
    resume = config["training"].get("resume", False)

    # 创建CSV文件并写入表头
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                "epoch", "train_loss", "val_loss",
                "resistance_auc", "resistance_precision", "resistance_f1",
                "mechanism_auc", "mechanism_precision", "mechanism_recall", "mechanism_f1",
                "mechanism_rare_f1",
                "antibiotic_auc", "antibiotic_precision", "antibiotic_recall", "antibiotic_f1",
                "antibiotic_rare_f1",
                "remove_auc", "remove_precision", "remove_f1"
            ]
            writer.writerow(header)

    # 梯度裁剪参数 (增加梯度裁剪强度)
    max_grad_norm = config["training"].get("max_grad_norm", 0.5)  # 减小最大梯度范数

    # 加载数据集
    train_dataset = torch.load("processed_data/train_dataset.pt")
    val_dataset = torch.load("processed_data/val_dataset.pt")

    num_mechanism = len(train_dataset[0]["mechanism_labels"])
    num_antibiotic = len(train_dataset[0]["antibiotic_labels"])

    # 打印类别分布统计
    for split_name, dataset in zip(["Train", "Val"], [train_dataset, val_dataset]):
        mechanism_counts = torch.zeros(num_mechanism)
        antibiotic_counts = torch.zeros(num_antibiotic)
        resistance_count = 0

        for sample in dataset:
            resistance_count += sample["is_arg"]
            if sample["is_arg"] == 1:
                mechanism_counts += sample["mechanism_labels"]
                antibiotic_counts += sample["antibiotic_labels"]

        print(
            f"{split_name} Resistance Positive: {resistance_count}/{len(dataset)} ({resistance_count / len(dataset):.2%})")
        print(f"{split_name} mechanism_cls_counts", mechanism_counts.int().tolist())
        print(f"{split_name} antibiotic_cls_counts", antibiotic_counts.int().tolist())

    # 计算稀有类别索引
    sorted_mech_indices = torch.argsort(mechanism_counts)
    rare_mech_indices = sorted_mech_indices[:int(num_mechanism * 0.25)].tolist()
    sorted_anti_indices = torch.argsort(antibiotic_counts)
    rare_anti_indices = sorted_anti_indices[:int(num_antibiotic * 0.25)].tolist()

    print(f"Rare mechanism indices: {rare_mech_indices}")
    print(f"Rare antibiotic indices: {rare_anti_indices}")

    # 创建带类别权重的采样器
    mechanism_weights = calculate_class_weights(train_dataset, num_mechanism, "mechanism")
    antibiotic_weights = calculate_class_weights(train_dataset, num_antibiotic, "antibiotic")

    sample_weights = []
    for sample in train_dataset:
        if sample["is_arg"] == 1:
            mech_weight = torch.mean(mechanism_weights[1:][sample["mechanism_labels"].bool()]).item()
            anti_weight = torch.mean(antibiotic_weights[1:][sample["antibiotic_labels"].bool()]).item()
            weight = (mech_weight + anti_weight) / 2
        else:
            weight = 1.0
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        pin_memory=True
    )

    # 模型配置
    model_config = AutoConfig.from_pretrained(config["model_1"]["name"])
    model_config.num_mechanism_labels = num_mechanism
    model_config.num_antibiotic_labels = num_antibiotic
    model_config.hidden_dropout_prob = config["regularization"].get("hidden_dropout", 0.3)
    model_config.attention_probs_dropout_prob = config["regularization"].get("attn_dropout", 0.2)
    model_config.classifier_dropout = config["regularization"].get("classifier_dropout", 0.4)

    # 添加稀有类别信息
    model_config.rare_mechanism_indices = rare_mech_indices
    model_config.rare_antibiotic_indices = rare_anti_indices

    # 加载模型
    model = GCM_MultiLabelModel.from_pretrained(
        config["model_1"]["name"],
        config=model_config
    ).to(device)

    # 创建优化器和学习率调度器 (减小初始学习率)
    optimizer = AdamW(
        [
            {"params": model.esm.parameters(), "lr": float(config["training"]["learning_rate"]) * 0.05},  # 减小ESM学习率
            {"params": [p for n, p in model.named_parameters() if "esm" not in n],
             "lr": float(config["training"]["learning_rate"]) * 0.8}  # 减小头部学习率
        ],
        weight_decay=config["training"].get("weight_decay", 0.01)
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * 1,
        num_training_steps=len(train_loader) * config["training"]["epochs"]
    )

    # 训练状态跟踪
    start_epoch = 0
    best_val_loss = float('inf')
    best_mechanism_f1 = 0.0
    best_rare_f1 = -1.0
    early_stop_counter = 0
    patience = config["training"].get("patience", 5)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"].get("fp16", False))

    # 恢复训练检查点
    if resume and os.path.isfile(checkpoint_path):
        start_epoch, best_val_loss, best_rare_f1, early_stop_counter = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler
        )
        start_epoch += 1
        print(f"=> 恢复训练: 从epoch {start_epoch}开始")
        print(
            f"=> 恢复状态: best_val_loss={best_val_loss:.4f}, best_rare_f1={best_rare_f1:.4f}, early_stop_counter={early_stop_counter}")

    # 学习率调度器 (添加学习率衰减)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        total_train_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']} - Training")

        # 第一阶段：冻结ESM主干
        if epoch < config["training"]["epochs"] // 3:
            for param in model.esm.parameters():
                param.requires_grad = False
        else:
            for param in model.esm.parameters():
                param.requires_grad = True

        # 监控梯度范数
        total_grad_norm = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            inputs = {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                "resistance_labels": batch["is_arg"].to(device, non_blocking=True),
                "mechanism_labels": batch["mechanism_labels"].to(device, non_blocking=True),
                "antibiotic_labels": batch["antibiotic_labels"].to(device, non_blocking=True),
                "remove_labels": batch["remove_label"].to(device, non_blocking=True),
            }

            optimizer.zero_grad()

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=config["training"].get("fp16", False)):
                outputs = model(**inputs)
                loss = outputs.loss if outputs.loss is not None else 0.0

            # 如果损失是NaN，跳过此批次
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ 检测到NaN/Inf损失，跳过此批次")
                continue  # 跳过后续步骤，进入下一个批次

            # 梯度缩放和裁剪
            scaler.scale(loss).backward()

            # 检查梯度是否存在NaN
            nan_detected = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    nan_detected = True
                    break

            if nan_detected:
                print("⚠️ 检测到NaN梯度，跳过此批次")
                # 清除当前批次的梯度
                optimizer.zero_grad()
                # 跳过优化步骤
                continue  # 跳过后续步骤，进入下一个批次

            scaler.unscale_(optimizer)

            # 梯度裁剪
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                    error_if_nonfinite=True
                )
            except RuntimeError as e:
                if 'non-finite' in str(e):
                    print("⚠️ 检测到非有限梯度，跳过此批次")
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

            # 记录梯度范数
            if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                total_grad_norm += grad_norm.item()
                num_batches += 1

            # 优化器更新
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

        # 计算平均梯度范数
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0
        print(f"平均梯度范数: {avg_grad_norm:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        all_resistance_preds, all_resistance_labels = [], []
        all_mechanism_preds, all_mechanism_labels = [], []
        all_antibiotic_preds, all_antibiotic_labels = [], []
        all_remove_preds, all_remove_labels = [], []
        rare_mechanism_preds, rare_mechanism_labels = [], []
        rare_antibiotic_preds, rare_antibiotic_labels = [], []

        print(f"Epoch {epoch + 1} - Validation")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                inputs = {
                    "input_ids": batch["input_ids"].to(device, non_blocking=True),
                    "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                    "resistance_labels": batch["is_arg"].to(device, non_blocking=True),
                    "mechanism_labels": batch["mechanism_labels"].to(device, non_blocking=True),
                    "antibiotic_labels": batch["antibiotic_labels"].to(device, non_blocking=True),
                    "remove_labels": batch["remove_label"].to(device, non_blocking=True),
                }

                outputs = model(**inputs)

                # 检查损失是否为NaN
                if outputs.loss is not None:
                    total_val_loss += outputs.loss.item()
                else:
                    print("⚠️ 验证损失为None，跳过此批次")
                    continue

                # 获取预测概率
                resistance_probs = torch.sigmoid(outputs.logits[0]).detach().cpu().numpy()
                mechanism_probs = torch.sigmoid(outputs.logits[1]).detach().cpu().numpy()
                antibiotic_probs = torch.sigmoid(outputs.logits[2]).detach().cpu().numpy()
                remove_probs = torch.sigmoid(outputs.logits[3]).detach().cpu().numpy()

                # 收集所有预测和标签
                all_resistance_preds.extend(resistance_probs)
                all_resistance_labels.extend(inputs["resistance_labels"].cpu().numpy())

                # 创建正样本掩码
                mask = inputs["resistance_labels"].cpu().numpy() == 1
                if mask.sum() > 0:
                    all_mechanism_preds.extend(mechanism_probs[mask])
                    all_mechanism_labels.extend(inputs["mechanism_labels"].cpu().numpy()[mask])
                    all_antibiotic_preds.extend(antibiotic_probs[mask])
                    all_antibiotic_labels.extend(inputs["antibiotic_labels"].cpu().numpy()[mask])
                    all_remove_preds.extend(remove_probs[mask])
                    all_remove_labels.extend(inputs["remove_labels"].cpu().numpy()[mask])

                    # 收集稀有类别的预测
                    if len(rare_mech_indices) > 0:
                        rare_mechanism_preds.append(mechanism_probs[mask][:, rare_mech_indices])
                        rare_mechanism_labels.append(
                            inputs["mechanism_labels"].cpu().numpy()[mask][:, rare_mech_indices])

                    if len(rare_anti_indices) > 0:
                        rare_antibiotic_preds.append(antibiotic_probs[mask][:, rare_anti_indices])
                        rare_antibiotic_labels.append(
                            inputs["antibiotic_labels"].cpu().numpy()[mask][:, rare_anti_indices])

        # 计算平均验证损失
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

        # 计算稀有类别的指标
        rare_mechanism_f1 = 0.0
        rare_antibiotic_f1 = 0.0

        if len(rare_mechanism_preds) > 0:
            rare_mech_preds = np.concatenate(rare_mechanism_preds, axis=0)
            rare_mech_labels = np.concatenate(rare_mechanism_labels, axis=0)
            rare_mech_metrics = compute_metrics(rare_mech_preds, rare_mech_labels, task_type="multilabel")
            rare_mechanism_f1 = rare_mech_metrics["f1_macro"]
        else:
            print("⚠️ No rare mechanism samples in validation set")

        if len(rare_antibiotic_preds) > 0:
            rare_anti_preds = np.concatenate(rare_antibiotic_preds, axis=0)
            rare_anti_labels = np.concatenate(rare_antibiotic_labels, axis=0)
            rare_anti_metrics = compute_metrics(rare_anti_preds, rare_anti_labels, task_type="multilabel")
            rare_antibiotic_f1 = rare_anti_metrics["f1_macro"]
        else:
            print("⚠️ No rare antibiotic samples in validation set")

        # 计算整体指标
        val_metrics = {
            "resistance": compute_metrics(np.array(all_resistance_preds), np.array(all_resistance_labels),
                                          task_type="binary"),
            "mechanism": compute_metrics(
                np.array(all_mechanism_preds) if len(all_mechanism_preds) > 0 else np.zeros((0, num_mechanism)),
                np.array(all_mechanism_labels), task_type="multilabel"),
            "antibiotic": compute_metrics(
                np.array(all_antibiotic_preds) if len(all_antibiotic_preds) > 0 else np.zeros((0, num_antibiotic)),
                np.array(all_antibiotic_labels), task_type="multilabel"),
            "remove": compute_metrics(np.array(all_remove_preds), np.array(all_remove_labels), task_type="binary"),
        }

        # 保存指标到CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                total_train_loss / len(train_loader) if len(train_loader) > 0 else float('nan'),
                avg_val_loss,
                val_metrics['resistance']['roc_auc'],
                val_metrics['resistance']['precision'],
                val_metrics['resistance']['f1'],
                val_metrics['mechanism']['roc_auc_macro'],
                val_metrics['mechanism']['precision_macro'],
                val_metrics['mechanism']['recall_macro'],
                val_metrics['mechanism']['f1_macro'],
                rare_mechanism_f1,
                val_metrics['antibiotic']['roc_auc_macro'],
                val_metrics['antibiotic']['precision_macro'],
                val_metrics['antibiotic']['recall_macro'],
                val_metrics['antibiotic']['f1_macro'],
                rare_antibiotic_f1,
                val_metrics['remove']['roc_auc'],
                val_metrics['remove']['precision'],
                val_metrics['remove']['f1']
            ])

        # 打印详细指标
        print(f"""
            Epoch {epoch + 1} | 
            Train Loss: {total_train_loss / len(train_loader):.4f} | 
            Val Loss: {avg_val_loss:.4f} 
            --- Val Metrics ---
            [Resistance] AUC: {val_metrics['resistance']['roc_auc']:.4f} | Precision: {val_metrics['resistance']['precision']:.4f} | F1: {val_metrics['resistance']['f1']:.4f}
            [Mechanism] Macro AUC: {val_metrics['mechanism']['roc_auc_macro']:.4f} | Precision: {val_metrics['mechanism']['precision_macro']:.4f} | Recall: {val_metrics['mechanism']['recall_macro']:.4f} | F1: {val_metrics['mechanism']['f1_macro']:.4f} | Rare F1: {rare_mechanism_f1:.4f} (n={len(all_mechanism_labels)})
            [Antibiotic] Macro AUC: {val_metrics['antibiotic']['roc_auc_macro']:.4f} | Precision: {val_metrics['antibiotic']['precision_macro']:.4f} | Recall: {val_metrics['antibiotic']['recall_macro']:.4f} | F1: {val_metrics['antibiotic']['f1_macro']:.4f} | Rare F1: {rare_antibiotic_f1:.4f} (n={len(all_antibiotic_labels)})
            [Remove] AUC: {val_metrics['remove']['roc_auc']:.4f} | Precision: {val_metrics['remove']['precision']:.4f} | F1: {val_metrics['remove']['f1']:.4f}
        """)

        # 更新学习率调度器
        lr_scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.2e}")

        # 早停和模型保存逻辑
        should_save = False
        current_rare_f1 = rare_mechanism_f1 + rare_antibiotic_f1

        # 监控稀有类别的F1分数
        if current_rare_f1 > best_rare_f1:
            best_rare_f1 = current_rare_f1
            should_save = True
            print(
                f"🔥 New best rare classes F1: Mechanism={rare_mechanism_f1:.4f}, Antibiotic={rare_antibiotic_f1:.4f}")

        # 监控整体验证损失
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            should_save = True
            early_stop_counter = 0
            print(f"🔥 New best Validation Loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"🚫 Validation loss increased for {early_stop_counter}/{patience} epochs")

        # 保存最佳模型
        if should_save:
            model.save_pretrained("outputs/AutoCNN_NewMoE_ASL_outputs_22/best_model")
            print(f"💾 Saved best model_1 at epoch {epoch + 1}")

        # 保存最新模型
        model.save_pretrained(f"outputs/AutoCNN_NewMoE_ASL_outputs_22/latest_model")

        # 保存检查点（每个epoch都保存）
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_rare_f1': best_rare_f1,
            'early_stop_counter': early_stop_counter
        }, checkpoint_dir)
        print(f"💾 Saved checkpoint for epoch {epoch + 1}")

        # 早停判断
        if early_stop_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()