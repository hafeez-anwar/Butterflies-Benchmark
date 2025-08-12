# 🖼️ Model Encoding & Classification Pipeline

This repository contains a **flexible, config-driven pipeline** for:
1. Extracting image features (encodings) using pre-trained CNN and Transformer models.
2. Running classification using a **Linear SVM** on the extracted encodings.
3. Supporting **three modes**:
   - **1** ➡ End-to-End Encoding + Classification
   - **2** ➡ Encoding Only
   - **3** ➡ Classification Only

The pipeline is designed for benchmarking multiple pre-trained models across datasets, with resume capability for long runs.

---

## 📂 Dataset
You can download the dataset from the following links:
1. *Dataset -1*: https://github.com/hafeez-anwar/FGBR
2. *Dataset -2*: https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species
After downloading, update the `dataset_path` in your `config.yaml` to point to the dataset location.

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/model-encoding-classification-pipeline.git
cd model-encoding-classification-pipeline
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🛠️ Configuration

All paths and parameters are controlled through **`config.yaml`**:

Example:
```yaml
dataset_path: "datasets/"
encodings_dir: "encodings/"
results_dir: "results/"
logging_level: "INFO"
device: "cuda"   # or "cpu"
resume: true

model_type: ["cnn", "transformer"]
cnn_models: ["resnet50", "efficientnet_b0"]
transformer_models: ["vit_base_patch16_224"]

kfold:
  n_splits: 5
  n_repeats: 1

random_state: 42
save_final_models: true
```

---

## 🚀 Usage

Run the main driver script with one of three modes:

### 1️⃣ End-to-End Encoding + Classification
```bash
python main.py --mode 1 --config config.yaml
```
or
```bash
python main.py --mode all
```

### 2️⃣ Encoding Only
```bash
python main.py --mode 2 --config config.yaml
```
or
```bash
python main.py --mode encode
```

### 3️⃣ Classification Only
```bash
python main.py --mode 3 --config config.yaml
```
or
```bash
python main.py --mode classify
```

---

## 📄 Scripts

- **`encode-images.py`** → Extracts features from all images using the models specified in `config.yaml`.
- **`classify.py`** → Runs Stratified/Repeated K-Fold SVM classification on the generated encodings.
- **`main.py`** → Orchestrates the workflow with three run modes.

---

## 📦 Outputs

- Encodings are saved in:
  ```
  encodings/<dataset>/<cnn|transformer>/<model>/
    model_encodings.npy
    model_labels.npy
  ```
- Classification results (CSV + optional Excel) in:
  ```
  results/<dataset>/<cnn|transformer>/
    model_kfold_metrics.csv
  results/summary_kfold_all_models.csv
  ```

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgements
- [PyTorch](https://pytorch.org/)
- [TIMM Models](https://huggingface.co/timm)
- [Scikit-learn](https://scikit-learn.org/)
