# PLM-ARG: Antibiotic Resistance Gene Identification using a pretrained protein language model
Antibiotic resistance genes (ARGs) are rapidly spreading and pose a serious global health threat. Existing computational tools often fail to detect novel ARGs or provide comprehensive functional annotation.To overcome these limitations, we propose CGC-ARG,a novel multi-task deep learning framework.
## 1. Training data collection

We constructed acomprehensive ARGdataset throughhierarchical integration of six authoritative databases. HMD-ARG served as the foundation due to its manual curation andclear classifcation. We systematically incorporated sequences CARD from fve additional databases: AMRFinderPlus,DeepARG-DB,MEGARes and ARG-ANNOT to ensurebroad coverage.The mergeddataset underwentstringentdeduplication using CD-HITat 100% sequence identity.Following manual curation across three dimensions antibiotic class, resistance mechanism, and gene mobility we obtained 19,980 high-quality sequences spanning 16 antibiotic classes,6 resistance mechanisms, and 2 mobility labels.
## 2. Model architecture
CGC-ARG employs a gated mixture-of-experts framework with two core components: feature extraction and multi-tasklearning.Protein sequences are processed through ESM-2 to extract contextual features, then combined with(AutoCNN) to a multi-scale convolutional attention module capture local structural patterns.
A hierarchical expert system with shared and task-specificlayers uses gating mechanisms for dynamic knowledge fusion.Four task-specifc heads simultaneously predict resistance status,antibiotic class,resistance mechanism and gene mobility. To address multi-label imbalance,we employ anasymmetric loss function with numerical stabilization for effective training.
## 3.Project Structure
In this project, the ARG folder contains my model files, training files, testing files, prediction files, as well as data files and related processing files. The baseline folder contains the comparison baseline model, and the ESM2 folder contains the model files needed for my model.
## 4.Dependencies



- python 3.9.0
- pytorch 1.13.1
- numpy 2.0.2
- pandas 2.2.3


