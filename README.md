# Document Country Detection using DiT + ArcFace
## Quick Start Guide

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
#### Add this to the install command in Environment Setup:
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```
### Dataset Preparation

Extract the provided dataset archive into the root directory of the project.

### Model Weights

Model weights are provided via a separate link and should be placed in the `best_model` directory. The provided weights were trained using the following parameters:
```json
{
  "num_epochs": 70,
  "lr_head": 0.0003,
  "lr_backbone": 3e-05,
  "weight_decay": 0.001,
  "total_steps": 2100,
  "warmup_steps": 252,
  "embed_dim": 512,
  "s": 15.0,
  "m": 0.15,
  "drop_p": 0.025
}
```

### Inference Example

Run inference on a single image:
```bash
python infer.py path/to/image.jpg
```

Interactive mode:
```bash
python infer.py
# Enter image paths when prompted
```

### Training

To train the model from scratch:
```bash
python train.py
```

Training checkpoints and metrics are saved automatically.



---
## Project Structure

```
.
├── best_model/
├── checkpoints/
├── dataset/
├── metrics/
├── model.py
├── infer.py
├── preprocess.py
├── train.py
├── Metrics.py
└── requirements.txt
```

## Project Overview

This project implements a machine learning model for determining the country of origin from images of identification documents (passports, ID cards, driver licenses, voter cards, etc.). The model is based on a DiT-base transformer architecture coupled with an ArcFace classification head, optimized to handle a variety of document layouts and formats without relying on the MRZ (Machine Readable Zone).

The DiT transformer backbone was selected for its superior capability to understand complex layouts, generalize across diverse document types, and scalability for future extensions. It is pretrained specifically on document-like images, making it highly effective for this task.

## Model Architecture

- **Backbone**: DiT-base Transformer (Encoder-only)
  - Pretrained model: `microsoft/dit-base`
  - Transformer encoder utilized for extracting robust embeddings
- **Pooling**: Adaptive average pooling followed by linear projection and L2 normalization
- **Head**: ArcFace (margin-based classification)
  - Ensures dense intra-class clusters and stable logits across hundreds of classes

## Key Features

- **Generalization**: Does not rely on MRZ zones, enabling broad applicability
- **Performance**: Achieves inference under 1 second per image on GPU without additional optimizations
- **Future-proof**: Scalable architecture for accommodating additional document types and countries

## Dataset and Preprocessing

The dataset contains synthetic passport images from multiple countries. The preprocessing pipeline includes:

- **Letterbox resizing** (maintaining aspect ratio)
- **Normalization** (ImageNet statistics)
- **Data Augmentation** (rotations, brightness adjustments, perspective distortion, and mild noise)

Planned future improvements:

- Acquiring additional datasets (currently restricted due to cost)
- Implementing cropping strategies (e.g., removing faces and MRZ zones)
- Training on partial images (inspired by methods like those in depth estimation models)

## Training and Evaluation

- **Optimizer**: AdamW with separate learning rates for backbone and classification head
- **Scheduler**: Cosine scheduler with warm-up
- **Gradient Accumulation**: Effective batch size increase
- **Early Stopping**: Implemented to prevent overfitting

### Metrics

- **Accuracy**:
  - Training: 76%
  - Validation: 91%
- **ROC-AUC**: 0.98 
- **t-SNE** visualizations and heatmaps generated to analyze embeddings

## Optimizations (Planned)

- **ONNX to TensorRT Conversion**: Expected 2-4x speedup
- **TorchScript Compilation**: Up to 1.4x faster inference
- **FP16 Quantization**: Approximately 40% memory reduction
- **INT8 Post-training Quantization**: Approximately 3x CPU speedup

These optimizations are planned enhancements for future iterations.

## Полное пояснение на русском языке

Данный проект реализует модель машинного обучения, предназначенную для определения страны происхождения по изображениям документов (паспорта, ID-карты, водительские удостоверения и др.). В основе модели используется архитектура трансформера DiT-base с классификационной головой ArcFace. Это решение позволяет обрабатывать документы с различными форматами и структурами без учёта MRZ-зоны.

### Подготовка датасета

Разархивируйте предоставленный датасет в корневой папке проекта.

### Веса модели

Файлы весов модели предоставляются отдельной ссылкой и должны быть размещены в папке `best_model`. Веса были получены при обучении модели с указанными выше параметрами.

### Запуск

Для запуска окружения и инференса используйте указанные выше команды в разделе "Quick Start Guide".

