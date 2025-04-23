# Federated Aggregation Method Comparison

This repository contains code and experiments from the paper:

**"On the Effectiveness of Federated Aggregation Methods Across Different Model Types"**  
Cole Feuer, Tyler Thraves, Sahithi Kamireddy

## 📄 Overview

Federated Learning (FL) allows collaborative training on distributed data without centralizing raw datasets. In this project, we empirically compare three aggregation methods across multiple model architectures and data modalities:

- **FedAvg**: Standard federated averaging of local weights.
- **FedLAMA**: Layer-wise adaptive model aggregation.
- **FedDist**: Output-level aggregation using soft-label distillation on a shared public dataset.

We evaluate each method on three distinct model types:

- Convolutional Neural Network (CNN) on CIFAR-10 (image classification)
- Multi-Layer Perceptron (MLP) on MIMIC-III (clinical tabular data)
- DistilBERT Transformer on Sentiment140 (text sentiment classification)

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `CNNAvg.ipynb` | CNN model training with FedAvg on CIFAR-10 |
| `CNNDist.ipynb` | CNN model training with FedDist on CIFAR-10 |
| `CNNLAMA.ipynb` | CNN model training with FedLAMA on CIFAR-10 |
| `FedAvg_mimicIII_MLP.ipynb` | MLP training on MIMIC-III with FedAvg |
| `FedDist_mimicIII_MLP.ipynb` | MLP training on MIMIC-III with FedDist |
| `FedLAMA_mimicIII_MLP.ipynb` | MLP training on MIMIC-III with FedLAMA |
| `TransformerAllMethods.py` | DistilBERT-based training on Sentiment140 for all aggregation methods |
| `Federated_Learning.pdf` | Paper describing the methodology and results |

## 🧪 Datasets

- **CIFAR-10** (Image Classification) — [Download](https://www.cs.toronto.edu/~kriz/cifar.html)
- **MIMIC-III** (Clinical Tabular Data) — [PhysioNet Access Required](https://physionet.org/content/mimiciii/1.4/)
- **Sentiment140** (Text Sentiment) — [Download](http://help.sentiment140.com/for-students/)
