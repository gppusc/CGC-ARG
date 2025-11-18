import csv
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig
import yaml
from tqdm import tqdm
from MOE import ESM2MultiLabel
from utils import compute_metrics
from Dataset import ProteinDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train():
    config = load_config("config.yaml")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    csv_file = "validation_metrics.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "epoch", "train_loss", "val_loss",
            "resistance_auc", "resistance_precision", "resistance_f1",
            "mechanism_auc", "mechanism_precision", "mechanism_recall", "mechanism_f1",
            "antibiotic_auc", "antibiotic_precision", "antibiotic_recall", "antibiotic_f1",
            "remove_auc", "remove_precision", "remove_f1"
        ]
        writer.writerow(header)

    # 梯度裁剪参数
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)

    # Load dataset
    train_dataset = torch.load("processed_data/train_dataset.pt")
    val_dataset = torch.load("processed_data/val_dataset.pt")
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    num_mechanism = len(train_dataset[0]["mechanism_labels"])
    num_antibiotic = len(train_dataset[0]["antibiotic_labels"])

    # 标签统计
    for split_name, dataset in zip(["Train", "Val"], [train_dataset, val_dataset]):
        mechanism_counts = torch.zeros(num_mechanism)
        antibiotic_counts = torch.zeros(num_antibiotic)
        for sample in dataset:
            if sample["is_arg"] == 1:
                mechanism_counts += sample["mechanism_labels"]
                antibiotic_counts += sample["antibiotic_labels"]
        print(f"{split_name} mechanism_cls_counts", mechanism_counts.int().tolist())
        print(f"{split_name} antibiotic_cls_counts", antibiotic_counts.int().tolist())

    # Model setup
    model_config = AutoConfig.from_pretrained(config["model_1"]["name"])
    model_config.num_mechanism_labels = num_mechanism
    model_config.num_antibiotic_labels = num_antibiotic
    model_config.hidden_dropout_prob = config["regularization"].get("hidden_dropout", 0.3)
    model_config.attention_probs_dropout_prob = config["regularization"].get("attn_dropout", 0.2)
    model_config.classifier_dropout = config["regularization"].get("classifier_dropout", 0.4)

    model = ESM2MultiLabel.from_pretrained(
        config["model_1"]["name"],
        config=model_config,
        filter_size=config["model_1"]["filter_size"]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * config["training"]["epochs"]
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 2

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_train_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']} - Training")
        for batch in tqdm(train_loader, desc="Training", leave=False):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "resistance_labels": batch["is_arg"].to(device),
                "mechanism_labels": batch["mechanism_labels"].to(device),
                "antibiotic_labels": batch["antibiotic_labels"].to(device),
                "remove_labels": batch["remove_label"].to(device),
            }
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        all_resistance_preds, all_resistance_labels = [], []
        all_mechanism_preds, all_mechanism_labels = [], []
        all_antibiotic_preds, all_antibiotic_labels = [], []
        all_remove_preds, all_remove_labels = [], []

        print(f"Epoch {epoch + 1} - Validation")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "resistance_labels": batch["is_arg"].to(device),
                    "mechanism_labels": batch["mechanism_labels"].to(device),
                    "antibiotic_labels": batch["antibiotic_labels"].to(device),
                    "remove_labels": batch["remove_label"].to(device),
                }
                outputs = model(**inputs)
                total_val_loss += outputs.loss.item()

                resistance_probs = torch.sigmoid(outputs.logits[0]).cpu().numpy()
                mechanism_probs = torch.sigmoid(outputs.logits[1]).cpu().numpy()
                antibiotic_probs = torch.sigmoid(outputs.logits[2]).cpu().numpy()
                remove_probs = torch.sigmoid(outputs.logits[3]).cpu().numpy()

                all_resistance_preds.extend(resistance_probs)
                all_resistance_labels.extend(inputs["resistance_labels"].cpu().numpy())

                mask = inputs["resistance_labels"].cpu().numpy() == 1
                if mask.sum() > 0:
                    all_mechanism_preds.extend(mechanism_probs[mask])
                    all_mechanism_labels.extend(inputs["mechanism_labels"].cpu().numpy()[mask])
                    all_antibiotic_preds.extend(antibiotic_probs[mask])
                    all_antibiotic_labels.extend(inputs["antibiotic_labels"].cpu().numpy()[mask])
                    all_remove_preds.extend(remove_probs[mask])
                    all_remove_labels.extend(inputs["remove_labels"].cpu().numpy()[mask])

        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics = {
            "resistance": compute_metrics(np.array(all_resistance_preds), np.array(all_resistance_labels), task_type="binary"),
            "mechanism": compute_metrics(
                np.array(all_mechanism_preds) if len(all_mechanism_preds) > 0 else np.zeros((0, num_mechanism)),
                np.array(all_mechanism_labels), task_type="multilabel"),
            "antibiotic": compute_metrics(
                np.array(all_antibiotic_preds) if len(all_antibiotic_preds) > 0 else np.zeros((0, num_antibiotic)),
                np.array(all_antibiotic_labels), task_type="multilabel"),
            "remove": compute_metrics(np.array(all_remove_preds), np.array(all_remove_labels), task_type="binary"),
        }

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                total_train_loss / len(train_loader),
                avg_val_loss,
                val_metrics['resistance']['roc_auc'],
                val_metrics['resistance']['precision'],
                val_metrics['resistance']['f1'],
                val_metrics['mechanism']['roc_auc_macro'],
                val_metrics['mechanism']['precision_macro'],
                val_metrics['mechanism']['recall_macro'],
                val_metrics['mechanism']['f1_macro'],
                val_metrics['antibiotic']['roc_auc_macro'],
                val_metrics['antibiotic']['precision_macro'],
                val_metrics['antibiotic']['recall_macro'],
                val_metrics['antibiotic']['f1_macro'],
                val_metrics['remove']['roc_auc'],
                val_metrics['remove']['precision'],
                val_metrics['remove']['f1']
            ])

        print(f"""
            Epoch {epoch + 1} | 
            Train Loss: {total_train_loss / len(train_loader):.4f} | 
            Val Loss: {avg_val_loss:.4f} 
            --- Val Metrics ---
            [Resistance] AUC: {val_metrics['resistance']['roc_auc']:.4f} | Precison: {val_metrics['resistance']['precision']:.4f} | f1: {val_metrics['resistance']['f1']:.4f}
            [Mechanism] Macro AUC: {val_metrics['mechanism']['roc_auc_macro']:.4f} | Macro Precision: {val_metrics['mechanism']['precision_macro']:.4f} | Macro Recall: {val_metrics['mechanism']['recall_macro']:.4f} | Macro F1: {val_metrics['mechanism']['f1_macro']:.4f} (n={len(all_mechanism_labels)})
            [Antibiotic] Macro AUC: {val_metrics['antibiotic']['roc_auc_macro']:.4f} | Macro Precision: {val_metrics['antibiotic']['precision_macro']:.4f} | Macro Recall: {val_metrics['antibiotic']['recall_macro']:.4f} | Macro F1: {val_metrics['antibiotic']['f1_macro']:.4f} (n={len(all_antibiotic_labels)})
            [Remove] AUC: {val_metrics['remove']['roc_auc']:.4f} | Precison: {val_metrics['remove']['precision']:.4f} | f1: {val_metrics['remove']['f1']:.4f}
        """)

        # 🟢 修改早停逻辑为监控机制 F1
        should_save = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            should_save = True
            early_stop_counter = 0
            print(f"🔥 New best Validation Loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"🚫 Validation loss increased for {early_stop_counter}/{patience} epochs")

        if should_save:
            model.save_pretrained("AutoCNN_Transformer_outputs_6/best_model")
            print(f"💾 Saved model_1 at epoch {epoch + 1}")

        if early_stop_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break

if __name__ == "__main__":
    train()
