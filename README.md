

# A facial foundation model for reproducible and scalable biomarker prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)]()
<!-- [![Paper](https://img.shields.io/badge/Nature%20Medicine-2025-orange.svg)]() -->

## 🔬 Overview
**FaceFound** is a facial foundation model that enables non-invasive, multi-system biomarker prediction from facial photographs.  
The model adopts a **progressive general-to-clinical pretraining strategy**, transferring knowledge from large-scale general image datasets to clinical biomarker prediction tasks across **62 biomarkers spanning 8 physiological systems**.

This repository provides:
1. **The core implementation of FaceFound**
2. **Pretrained weights**
3. **Analysis pipline of this paper**


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
git clone https://github.com/<YourOrg>/FaceFound.git
cd FaceFound
# For pretraining and fine-tuning
mamba env create -f PretrainingEnvironment.yml
# For paper analysis
mamba env create -f PaperEnvironment.yml
```

**Dependencies:**

For Python packages
- Python ≥ 3.10
- R ≥ 4.1
```
mamba create -n FaceFound python=3.10 pandas numpy seaborn scipy statsmodels plotnine pingouin 
mamba config --add channels r
mamba install  r-essentials r-base r-relaimpo 

# dcurves 
```

## 🧩 Model Training and fine-tuning

### 1. Pretraining (Self-supervised)

```bash
python train_pretext.py --config configs/pretext.yaml
```

### 2. Fine-tuning on Biomarkers

```bash
python train_biomarker.py --config configs/finetune.yaml
```




---

## 📊 Reproducing Results
**First**, download the [Predictions and Labels](xxxx)

> For `config.py`, it records the necessary and basic variables for the analysis

**Second**, Run the following scripts to reproduce the results in the paper:

- `01-Table.ipynb` Generate Table 1
- `02-ModelPerformance.ipynb` Evaluate model performance
- `03-DiseaseRiskEvaluationModel` Evaluate the clinical utility of the model
- `04-APP-Evaluation.ipynb` Evaluate the performance with predictions from the app of selfies
- `05-GeneticsEvaluation.ipynb` Evaluate the performance of FaceFound vs. PRS from the external cohorts with genotype data
- `06.x-Partx` Plot the results of the paper 
- `07-SupplementaryTable.ipynb` Generate Supplementary Table of results used in the paper


## 📚 Citation

If you use this repository, please cite our work:

<!-- ```
@article{FaceFound2025,
  title   = {FaceFound: A Facial Foundation Model for Multi-System Biomarker Prediction},
  author  = {Your Name et al.},
  journal = {Nature Medicine},
  year    = {2025},
  doi     = {10.1038/s41591-XXXX-XXXXX}
}
``` -->



## 🧾 License

This project is released under the [MIT License](LICENSE).


## 📨 Contact

For questions or collaboration, please contact:
 **[xutingfeng@big.ac.cn](mailto:xutingfeng@big.ac.cn)**

