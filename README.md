# Missing-aware Prompt for Robust Multimodal Perception under Missing Modalities

> Robust multimodal learning with missing-aware prompts and late fusion architecture.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

# 📖 Overview

This project focuses on **robust multimodal perception under missing modality scenarios**.

In real-world multimodal systems, image or text modalities are often incomplete due to noise, transmission failure, privacy restrictions, or data corruption. Traditional multimodal models usually assume complete multimodal input, which leads to severe performance degradation when modalities are missing.

To address this issue, this project proposes:

- **Missing-aware Prompt**
- **Late Fusion Architecture**
- **Robust Multimodal Learning Framework**

The proposed method explicitly models modality missing states and enables the model to dynamically adapt cross-modal fusion strategies under incomplete input conditions.

---

# ✨ Features

- 🔥 Missing-aware Prompt for modality missing perception
- 🧠 ResNet + BERT dual-encoder architecture
- 🌉 Late Fusion based multimodal interaction
- 📉 Robust learning under different missing rates
- 🧪 Full Fine-tuning vs Frozen Encoder experiments
- 📊 Support for MM-IMDb and Hateful Memes datasets
- ⚡ PyTorch implementation

---

# 🏗️ Method

## Overall Framework

```text
Image → ResNet ┐
               ├── Feature Fusion ──→ Classifier
Text  → BERT   ┘
         ↑
Missing-aware Prompt
```

The model introduces explicit prompts to indicate:

- image missing
- text missing
- complete modalities

This allows the network to distinguish:

- low-information input
- actual modality missing

thus improving robustness under incomplete multimodal conditions.

---

# 📂 Project Structure

```text
.
├── datasets/              # Dataset loading
├── models/                # Model architectures
│   ├── image_encoder.py
│   ├── text_encoder.py
│   ├── fusion.py
│   └── prompt.py
├── train.py               # Training script
├── test.py                # Evaluation script
├── utils/                 # Utility functions
├── records/               # Training logs
├── requirements.txt
└── README.md
```

---

# 📊 Datasets

## MM-IMDb

A multimodal multi-label movie genre classification dataset containing:

- movie posters
- plot text descriptions

Used to evaluate robustness under multimodal missing conditions.

## Hateful Memes

A multimodal meme classification dataset proposed by Meta AI.

The task requires joint understanding of:

- image semantics
- text semantics

which makes it highly suitable for evaluating cross-modal reasoning under missing modalities.

---

# ⚙️ Environment

## Requirements

- Python >= 3.10
- CUDA >= 11.8
- PyTorch >= 2.0

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

## Full Fine-tuning

```bash
python train.py \
    --dataset imdb \
    --missing_rate 0.2 \
    --epochs 10 \
    --lr 1e-4
```

## Frozen Encoder Training

```bash
python train.py \
    --dataset imdb \
    --missing_rate 0.2 \
    --freeze_encoder \
    --epochs 40 \
    --lr 1e-4
```

---

# 🧪 Experiments

Experiments were conducted on:

- NVIDIA RTX 4080
- Batch Size = 32
- Learning Rate = 1e-4

Missing rates:

```text
0 / 0.2 / 0.4 / 0.6 / 0.8
```

Evaluation metrics:

## MM-IMDb

- Subset Accuracy
- Hamming Accuracy
- Micro-F1
- Macro-F1

## Hateful Memes

- F1-score

---

# 📈 Main Findings

- Missing-aware Prompt improves robustness under missing modality conditions.
- Late Fusion maintains relatively stable semantic representations.
- Moderate missing rates may provide regularization effects.
- Frozen Encoder reduces trainable parameters but harms Macro-F1 performance.
- Prompt still helps cross-modal alignment under frozen feature space.

---

# 📚 Related Works

This project is inspired by recent studies on:

- missing modality learning
- prompt learning
- parameter-efficient multimodal adaptation

Relevant works include:

- SMIL
- Missing-aware Prompts
- Robust Multimodal Learning with Missing Modalities
- Prompt-based multimodal adaptation

---

# 🔬 Citation

If you find this project useful, please consider citing:

```bibtex
@article{yourname2026missingaware,
  title={Robust Multimodal Perception under Missing Modalities with Missing-aware Prompt},
  author={Your Name},
  year={2026}
}
```

---

# 📝 TODO

- [ ] Add CLIP encoder support
- [ ] Add LoRA fine-tuning
- [ ] Add dynamic prompt generation
- [ ] Add visualization tools
- [ ] Add missing modality reconstruction experiments

---

# 📄 License

This project is released under the MIT License.

---

# 🙏 Acknowledgements

Thanks to the open-source community and previous works on:

- multimodal learning
- prompt learning
- robust perception
- missing modality adaptation

for providing inspiration for this project.
