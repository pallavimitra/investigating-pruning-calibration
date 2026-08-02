



# Investigating Calibration and Corruption Robustness of Post-hoc Pruned Perception CNNs
Research code for analyzing the effects of post-hoc pruning on calibration, robustness, and efficiency of deep neural networks.

 **Paper:** https://openaccess.thecvf.com/content/CVPR2024W/SAIAD/papers/Mitra_Investigating_Calibration_and_Corruption_Robustness_of_Post-hoc_Pruned_Perception_CNNs_CVPRW_2024_paper.pdf

---

# Overview

Deep neural networks are increasingly deployed in safety-critical applications where computational efficiency and reliable uncertainty estimation are equally important. While neural network pruning is widely adopted to reduce computational cost, its impact on uncertainty calibration and robustness under distribution shift remains insufficiently understood.

This repository provides the official implementation accompanying our CVPR 2024 Workshop paper, which systematically investigates how different post-hoc pruning techniques influence:

- Classification accuracy
- Uncertainty calibration
- Natural corruption robustness
- Model efficiency

using CIFAR-10 and CIFAR-100 benchmarks.

---

# Pruning Methods

The repository evaluates three commonly used pruning paradigms:

### 1. Unstructured Weight-Level Pruning
Based on:

> Learning both Weights and Connections for Efficient Neural Networks

https://arxiv.org/abs/1506.02626

---

### 2. L1-Norm Filter Pruning

Based on:

> Pruning Filters for Efficient ConvNets

https://arxiv.org/abs/1608.08710

---

### 3. Channel Pruning (Network Slimming)

Based on:

> Network Slimming

https://arxiv.org/abs/1708.06519

---

The pruning implementations are adapted from:

**Rethinking the Value of Network Pruning**

https://github.com/Eric-mingjie/rethinking-network-pruning

---

# Repository Structure

```
weight-level/
channel-pruning/
L1_norm_pruning/
```

Each folder contains the complete pipeline for one pruning strategy.

---

# Experimental Pipeline

Every pruning method follows the same three-stage workflow:

```
Train
      ↓
Prune
      ↓
Fine-tune
      ↓
Evaluate
```

The repository additionally evaluates

- Expected Calibration Error (ECE)
- Robustness under natural corruptions

---

# Installation

Clone the repository

```bash
git clone https://github.com/pallavimitra/pruning-calibration-robustness.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Datasets

Experiments were conducted on

- CIFAR-10
- CIFAR-100
- CIFAR-10-C
- CIFAR-100-C

---

# Training

## Weight-Level Pruning

```bash
python cifar.py --dataset cifar10 --arch preresnet --depth 110
```

## L1 Filter Pruning

```bash
python main.py --dataset cifar10 --arch resnet --depth 110
```

## Channel Pruning

```bash
python main.py --dataset cifar10 --arch resnet --depth 164
```

---

# Pruning



---

# Fine-tuning



---

# Evaluation

The repository supports evaluation of

- Classification Accuracy
- Expected Calibration Error (ECE)
- Natural Corruption Robustness

using the provided evaluation scripts.

---

# Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Mitra_2024_CVPRW,
    author = {Mitra, Pallavi and ...},
    title = {Investigating Calibration and Corruption Robustness of Post-hoc Pruned Perception CNNs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year = {2024}
}
```

---

# License

MIT License

---

# Acknowledgements

This work was presented at the **CVPR 2024 Workshop on Safe AI for Automated Driving (SAIAD)**.
