# PLM-ARG: Antibiotic Resistance Gene Identification using a pretrained protein language model

This project implements a multi-task classification system based on the ESM protein language model, capable of simultaneously predicting:

🧬 Antimicrobial Resistance Gene (ARG) identification

⚙️ Resistance mechanism classification

💊 Antibiotic class classification

🗑️ Removal label prediction

## 📋 Training data collection

We constructed acomprehensive ARGdataset throughhierarchical integration of six authoritative databases. HMD-ARG served as the foundation due to its manual curation andclear classifcation. We systematically incorporated sequences CARD from fve additional databases: AMRFinderPlus,DeepARG-DB,MEGARes and ARG-ANNOT to ensurebroad coverage.The mergeddataset underwentstringentdeduplication using CD-HITat 100% sequence identity.Following manual curation across three dimensions antibiotic class, resistance mechanism, and gene mobility we obtained 19,980 high-quality sequences spanning 16 antibiotic classes,6 resistance mechanisms, and 2 mobility labels.

## 🏗️Model architecture

CGC-ARG employs a gated mixture-of-experts framework with two core components: feature extraction and multi-tasklearning.Protein sequences are processed through ESM-2 to extract contextual features, then combined with(AutoCNN) to a multi-scale convolutional attention module capture local structural patterns.
A hierarchical expert system with shared and task-specificlayers uses gating mechanisms for dynamic knowledge fusion.Four task-specifc heads simultaneously predict resistance status,antibiotic class,resistance mechanism and gene mobility. To address multi-label imbalance,we employ anasymmetric loss function with numerical stabilization for effective training.

## 🚀Code usage

### 1.Data Processing

🎯 Functions

- Data cleaning and standardization
- Multi-label encoding
- Dataset splitting
- Rare label handling
- Label coverage assurance

```bash
python Dataset.py
```

📥 Input Files

- myarg.csv - Positive sample dataset
- final_negative_dataset.csv - Negative sample dataset
  📤 Output Files
- train.csv - Training set
- val.csv - Validation set
- test.csv - Test set
- labels.py - Label definition file
  ⚙️ Key Parameters

```pyhton
dc_threshold = 50    # Rare label threshold for antibiotic classes
rm_threshold = 50    # Rare label threshold for resistance mechanisms
# Training/Validation/Test set ratio: 70%/15%/15%
```

###2. Dataset Processing
⚙️ Configuration Parameters

```pyhton
# Modify the following paths in the code
model_path = "../ESM2_t30_150M_UR50D"  # ESM model path
max_length = 1024  # Maximum sequence length
mask_ratio = 0.05  # Mask ratio during training
```

📤 Output Files

- processed_data/train_dataset.pt - Processed training set
- processed_data/val_dataset.pt - Processed validation set
- processed_data/test_dataset.pt - Processed test set
- Class Weight Files:
- pos_weight_mech.pt
- pos_weight_anti.pt
- pos_weight_remove.pt

### 3.Multi-task Model

🔧 Core Components

- AutoCNNLayer: Multi-scale convolutional feature extraction
- FeatureFusion: Gated feature fusion mechanism
- ExpertLayer: Task-specific expert networks
- DiversityGating: Diversity gating mechanism
- AntibioticAdapter: Antibiotic task adapter
  📉 Loss Functions
- AsymmetricLoss: Asymmetric loss for multi-label imbalanced data
- Dynamic weight adjustment
- Diversity regularization
  🚀 Usage

```pyhton
from AutoCNN_NewMoE_ASL_1 import GCM_MultiLabelModel
# Model configuration
config = {
    "num_mechanism_labels": 8, # Adjust according to labels.py
    "num_antibiotic_labels": 48,# Adjust according to labels.py
    "use_remove": True
}
model = GCM_MultiLabelModel(config)
```

🔄 Complete Workflow
Step 1: Data Preparation

```bash
# 1. Data preprocessing
python data_processing.py
# 2. Dataset generation  
python Dataset.py
```

Step 2: Model Training
Create a training script with the following pseudo-code:

```pyhton
from AutoCNN_NewMoE_ASL_1 import GCM_MultiLabelModel
from Dataset import preprocess_data
# Load data
train_dataset = torch.load('processed_data/train_dataset.pt')
# Load model and training loop...
```

Step 3: Model Inference
Use the trained model for prediction.

## 📋 Environment Requirements

```pyhton
python=3.9.0
pytorch=1.13.1
numpy=2.0.2
pandas=2.2.3
```

⚠️ Important Notes
Model Path: Ensure the ESM pre-trained model path is correc
Label Count: Adjust the number of labels according to actual data
Long Sequence Handling: Long sequences are automatically processed with sliding windows
Data Augmentation: Random mask augmentation is enabled during training, disabled during validation/testing
Class Weights: Automatically calculated to handle data imbalance
