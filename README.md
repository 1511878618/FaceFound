# A facial foundation model for reproducible and scalable biomarker prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)]()
<!-- [![Paper](https://img.shields.io/badge/Nature%20Medicine-2025-orange.svg)]() -->

## 🔬 Overview
**FaceFound** is a facial foundation model that enables non-invasive, multi-system biomarker prediction from facial photographs.  
The model adopts a **progressive general-to-clinical pretraining strategy**, transferring knowledge from large-scale general image datasets to clinical biomarker prediction tasks across **62 biomarkers spanning 8 physiological systems**.

This repository provides:
1. **The pretraining codes of FaceFound**
1. **Analysis pipline of this paper**


## 📄 Abstract
Most clinical biomarkers require invasive sampling and laboratory testing, limiting accessibility for large-scale screening and continuous monitoring.  
FaceFound leverages facial photographs—a ubiquitous and non-invasive data source—to estimate systemic biomarker levels with clinical-grade accuracy.  
Trained through progressive self-supervised and clinical fine-tuning, FaceFound outperformed baseline architectures (ResNet18 and Swin-Large), demonstrated reproducibility across internal and external cohorts, and showed **superior label efficiency** and **real-world deployability** via a smartphone application.


<!-- ## 🧠 Key Features
- **Progressive pretraining:** from general image datasets to clinical fine-tuning.  
- **Multi-biomarker prediction:** 62 biomarkers covering cardiovascular, metabolic, renal, and hematologic systems.  
- **Superior label efficiency:** maintains predictive power with as few as 400 samples.  
- **Cross-cohort validation:** evaluated on four external cohorts (AZ-EV1, AZ-EV2, AZTZ-EV, DX-EV).  
- **Deployed application:** smartphone-ready model supporting real-time biomarker estimation.   -->


## ⚙️ Installation

```bash
git clone git@github.com:1511878618/FaceFound.git
```

**Dependencies:**

For Python packages
- Python ≥ 3.10
- R ≥ 4.1
```
conda create -n FaceFound python=3.10 pandas numpy seaborn scipy statsmodels plotnine pingouin 
conda config --add channels r
conda install  r-essentials r-base r-relaimpo 
```

For pretraining
```bash
conda create -n FaceFoundPretraining python=3.10
pip install -r PretrainingRequirements.txt
```

## 🧩 Model Training and fine-tuning

### 1. Pretraining (Self-supervised)

```bash
bash FaceFound/scripts/pretrain.sh
```

### 2. Fine-tuning on Biomarkers

```bash
bash FaceFound/scripts/finetune.sh
```


## 📦Data Access

To replicated our results from this repo, you need to request access to the data.
 Please email  **[xutingfeng@big.ac.cn](mailto:xutingfeng@big.ac.cn)**
 to request access:
 
1. Provide your name, affiliation, intended use, and the specific files you need.

1. Data will be shared only in compliance with applicable ethics approvals and data-use agreements (e.g., de-identified outputs where appropriate).

## 📊 Analysis 

Run the following scripts to reproduce the results in the paper:

- `01-Table.ipynb` Generate Table 1
- `02-ModelPerformance.ipynb` Evaluate model performance
- `03-DiseaseRiskEvaluationModel` Evaluate the clinical utility of the model
- `04-APP-Evaluation.ipynb` Evaluate the performance with predictions from the app of selfies
- `05-GeneticsEvaluation.ipynb` Evaluate the performance of FaceFound vs. PRS from the external cohorts with genotype data
- `06.x-Partx` Plot the results of the paper 
- `07-SupplementaryTable.ipynb` Generate Supplementary Table of results used in the paper


## 📚 Citation

If you use this repository, please cite our work: 
Minxian Wang, Tingfeng Xu, Huixuan Xu et al. A facial foundation model for multi-system biomarker and disease risk prediction with real-world mobile deployment, 20 January 2026, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-8110055/v1]


## 🧾 License

This project is released under the [MIT License](LICENSE).


## 📨 Contact

For questions or collaboration, please contact:
 **[xutingfeng@big.ac.cn](mailto:xutingfeng@big.ac.cn)**

