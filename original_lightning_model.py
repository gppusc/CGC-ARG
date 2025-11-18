"""Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch.nn.functional as F
from torch.ao.ns.fx.utils import compute_cosine_similarity

from anomalib.models.components import AnomalyModule
from anomalib.models.patchcore.torch_model import PatchcoreModel

logger = logging.getLogger(__name__)


class Patchcore(AnomalyModule):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        input_size (tuple[int, int]): Size of the model input.
        backbone (str): Backbone CNN network
        layers (list[str]): Layers to extract features from the backbone CNN
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        coreset_sampling_ratio (float, optional): Coreset sampling ratio to subsample embedding.
            Defaults to 0.1.
        num_neighbors (int, optional): Number of nearest neighbors. Defaults to 9.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        layers: list[str],
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()

        self.model: PatchcoreModel = PatchcoreModel(
            input_size=input_size,
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[Tensor] = []
        self.threshold = 1.0

    def configure_optimizers(self) -> None:
        """Configure optimizers.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return None

    # Define a method to compute cosine similarity between a new embedding and a list of embeddings.
    # def compute_cosine_similarity(new_embedding, embeddings):
    #     print(f"new_embedding shape: {new_embedding.shape}")
    #     print(f"embeddings shape: {embeddings.shape}")
    #     # Normalize the embeddings
    #     new_embedding = new_embedding / (torch.norm(new_embedding, dim=1, keepdim=True) + 1e-10)
    #     embeddings = embeddings / (torch.norm(embeddings, dim=1, keepdim=True) + 1e-10)
    #
    #     # Compute cosine similarity
    #     similarity = torch.mm(new_embedding, embeddings.transpose(0, 1))
    #
    #     return similarity

    def compute_mean_std_similarity(self,embeddings):
        # 将列表中的每个二维特征图展平为一维
        flattened_embeddings = [embedding.flatten() for embedding in embeddings]
        # 将展平后的特征向量堆叠成一个新的PyTorch张量
        embeddings_tensor = torch.stack(flattened_embeddings)
        # 计算相似度矩阵
        normalized_embeddings = embeddings_tensor / (embeddings_tensor.norm(dim=1, keepdim=True) + 1e-10)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.transpose(0, 1))

        # similarity_matrix = torch.mm(embeddings_tensor, embeddings_tensor.transpose(0, 1))
        # 获取上三角矩阵，避免重复计算和自身相似度的影响
        upper_tri_similarity = similarity_matrix.triu(diagonal=1)
        # 计算均值和标准差
        mean_similarity = upper_tri_similarity[upper_tri_similarity != 0].mean()
        std_similarity = upper_tri_similarity[upper_tri_similarity != 0].std()
        return mean_similarity, std_similarity


    def update_threshold(self , embeddings):
        mean_similarity, std_similarity = self.compute_mean_std_similarity(embeddings)
        # 设定阈值为均值加标准差
        # print("std_similarity is ",std_similarity)
        # self.threshold = mean_similarity + std_similarity
        self.threshold = mean_similarity
        return self.threshold


    # Modified training_step method
    def training_step(self, batch, *args, **kwargs):
        del args, kwargs  # These variables are not used.

        self.model.feature_extractor.eval()
        embedding = self.model(batch["image"]).squeeze(0)

        # If self.embeddings is empty, just append the new embedding
        if len(self.embeddings) < 20:
            self.embeddings.append(embedding)
            # print("embedding is ",embedding.shape)
            # print(f"添加一个特征图进去，现在的特征图数量: {len(self.embeddings)}")
        else:
            # Compute cosine similarity
            embeddings_tensor = torch.stack(self.embeddings)
            #
            # print("embedding shape:", embedding.unsqueeze(0).shape)
            # print("embeddings_tensor shape:", embeddings_tensor.shape)
            expanded_embedding = embedding.expand_as(embeddings_tensor)
            similarity = compute_cosine_similarity(expanded_embedding, embeddings_tensor)
            # print("similarity is ",similarity)
            # print("threshold is ",self.threshold)
            # Check if the similarity of the new embedding with all existing embeddings is below the threshold
            if torch.all(similarity < self.threshold):
                self.embeddings.append(embedding)
                # similarity_value = similarity.item()
            #     print(f"添加一个特征图进去，现在的特征图数量: {len(self.embeddings)}")
            # else:
            #     print("与memory bank 特征相似")
            if len(self.embeddings) % 5 == 0:
                self.threshold = self.update_threshold(self.embeddings)
                # print(f"更新阈值为: {self.threshold:.4f}")



    def on_validation_start(self) -> None:
        """Apply subsampling to the embedding collected from the training set."""
        # NOTE: Previous anomalib versions fit subsampling at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("kipping the coreset subsampling.")
        # self.model.subsample_embedding(embeddings, self.coreset_sampling_ratio)
        self.model.memory_bank = embeddings

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename,
                image, label and mask

        Returns:
            dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        output = self.model(batch["image"])

        # Add anomaly maps and predicted scores to the batch.
        batch["anomaly_maps"] = output["anomaly_map"]
        batch["pred_scores"] = output["pred_score"]

        return batch


class PatchcoreLightning(Patchcore):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams) -> None:
        super().__init__(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            layers=hparams.model.layers,
            pre_trained=hparams.model.pre_trained,
            coreset_sampling_ratio=hparams.model.coreset_sampling_ratio,
            num_neighbors=hparams.model.num_neighbors,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
