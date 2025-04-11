

# GAMOS: Game-Theoretic Adaptive Minority Oversampling Strategy
Official implementation of GAMOS: a game-theoretic oversampling method for imbalanced classification problems.


**GAMOS** (Game-Theoretic Adaptive Minority Oversampling Strategy) is a novel oversampling algorithm that reframes the data generation process for imbalanced classification as a cooperative game among minority instances. It combines game-theoretic reasoning, adaptive strategy optimization, and feature-weighted noise injection to produce high-quality, diverse, and classifier-friendly synthetic samples.

This repository provides the official implementation of GAMOS, as described in the paper:

> **Türkoğlu, B. (2025).** * A Game-Theoretic Adaptive Oversampling Strategy for Imbalanced Data Classification*. (preparing for submission).

---

## 🔍 Features

- ✅ Game-theoretic payoff modeling  
- ✅ Adaptive interpolation strategy using directional learning  
- ✅ Feature-weighted Gaussian noise injection  
- ✅ Multi-criteria validation (detectability, classifier degradation, diversity)  
- ✅ Post-generation filtering  
- ✅ Seamless integration with `scikit-learn` pipelines  


---

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/GAMOS-oversampling.git
cd GAMOS-oversampling
pip install -r requirements.txt


@article{turkoglu2025gamos,
  author    = {Bahaeddin Türkoğlu},
  title     = { A Game-Theoretic Adaptive Oversampling Strategy for Imbalanced Data Classification},
  journal   = {preparing for submission},
  year      = {2025}
}
