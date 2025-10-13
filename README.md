

# FaceFound: A Facial Foundation Model for Multi-System Biomarker Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)]()
[![Paper](https://img.shields.io/badge/Nature%20Medicine-2025-orange.svg)]()

## 🔬 Overview
**FaceFound** is a facial foundation model that enables non-invasive, multi-system biomarker prediction from facial photographs.  
The model adopts a **progressive general-to-clinical pretraining strategy**, transferring knowledge from large-scale general image datasets to clinical biomarker prediction tasks across **62 biomarkers spanning 8 physiological systems**.

This repository provides the core implementation, pretrained weights, and evaluation scripts used in our study.

---

## 📄 Abstract
Most clinical biomarkers require invasive sampling and laboratory testing, limiting accessibility for large-scale screening and continuous monitoring.  
FaceFound leverages facial photographs—a ubiquitous and non-invasive data source—to estimate systemic biomarker levels with clinical-grade accuracy.  
Trained through progressive self-supervised and clinical fine-tuning, FaceFound outperformed baseline architectures (ResNet18 and Swin-Large), demonstrated reproducibility across internal and external cohorts, and showed **superior label efficiency** and **real-world deployability** via a smartphone application.

---

## 🧠 Key Features
- **Progressive pretraining:** from general image datasets to clinical fine-tuning.  
- **Multi-biomarker prediction:** 62 biomarkers covering cardiovascular, metabolic, renal, and hematologic systems.  
- **Superior label efficiency:** maintains predictive power with as few as 400 samples.  
- **Cross-cohort validation:** evaluated on four external cohorts (AZ-EV1, AZ-EV2, AZTZ-EV, DX-EV).  
- **Deployed application:** smartphone-ready model supporting real-time biomarker estimation.  

---

## ⚙️ Installation
```bash
git clone https://github.com/<YourOrg>/FaceFound.git
cd FaceFound
pip install -r requirements.txt
````

**Dependencies:**

* Python ≥ 3.10
* PyTorch ≥ 2.2
* OpenCV, NumPy, pandas, scikit-learn
* timm, albumentations, matplotlib

---

## 🧩 Model Training

### 1. Pretraining (Self-supervised)

```bash
python train_pretext.py --config configs/pretext.yaml
```

### 2. Fine-tuning on Biomarkers

```bash
python train_biomarker.py --config configs/finetune.yaml
```

### 3. Evaluation

```bash
python evaluate.py --checkpoint checkpoints/facefound_biomarkers.pth
```

---

## 📊 Reproducing Results

Scripts for reproducing main and supplementary results:

* `notebooks/Fig3_performance.ipynb` — model performance across biomarkers
* `notebooks/Fig5_label_efficiency.ipynb` — label efficiency analysis
* `notebooks/ExtendedData8_gradcam.ipynb` — model interpretation via Grad-CAM

---

## 📱 Deployment

FaceFound is implemented as a WeChat-based mobile application for real-time biomarker estimation.
The app frontend uses **WeChat native components** and **Vant Weapp**, while the backend is built with **Golang (Gin)**, **MySQL**, and **Redis**, deployed on **Ubuntu 22.04** with HTTPS and JWT-based authentication.

---

## 📚 Citation

If you use this repository, please cite our work:

```
@article{FaceFound2025,
  title   = {FaceFound: A Facial Foundation Model for Multi-System Biomarker Prediction},
  author  = {Your Name et al.},
  journal = {Nature Medicine},
  year    = {2025},
  doi     = {10.1038/s41591-XXXX-XXXXX}
}
```

---

## 🧾 License

This project is released under the [MIT License](LICENSE).

---

## 📨 Contact

For questions or collaboration, please contact:
📧 **[your.email@institution.edu](mailto:your.email@institution.edu)**
🌐 \[Project Page / Dataset Portal if available]

