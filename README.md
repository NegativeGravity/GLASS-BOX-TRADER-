# Glass Box Trader: Market Microstructure & Transformer Execution System

This repository contains the end-to-end pipeline for high-frequency alpha research and execution modeling. Built specifically for Kaggle-scale 1-minute cryptocurrency tick data, the system leverages explainable AI (XAI) principles to map nonlinear market dynamics using temporal attention mechanisms.

By bridging robust mathematical market auditing with deep sequence modeling, this architecture moves beyond black-box predictions, exposing the underlying reasoning of the Transformer across time horizons.

---

## Architecture Overview

The pipeline is strictly modular, dividing raw market microstructure into clean, stationarity-audited signals, and ultimately feeding a custom multi-head attention network.

* **Phase 1-3: Market Auditing & First Principles** - Raw tick ingestion, robust winsorization, and volatility-adjusted logarithmic return mapping.
* **Phase 4 & 7: Production Feature Engine** - Accelerated extraction of technical and momentum features utilizing standard optimized libraries (`ta`) to ensure production velocity, eliminate manual implementation overhead, and guarantee production-grade coding standards.
* **Phase 5: Sequence Generation** - PyTorch-native temporal windowing (60-minute lookbacks) optimized for memory-safe batch loading.
* **Phase 6-8: Explainable Deep Learning** - A custom `GlassBoxTransformer` engineered to extract, monitor, and visualize internal self-attention weights mapped directly to price action.

---

## Mathematical Formulation

### 1. Target Definition
The execution horizon is defined as a 15-minute forward window. The model targets binary classification of forward returns, predicting directional alpha based on current market state:

* R_(t, t+15) = (P_(t+15) / P_t) - 1
* Y_t = 1 if R_(t,t+15) > 0 else 0

### 2. Market Stationarity & Auditing
To handle the heavy-tailed nature of crypto market microstructure, strict winsorization is applied to logarithmic returns at a 5.0-sigma threshold. 

* r_t = ln(P_t / P_(t-1))
* r*_t = max(mu - 5*sigma, min(r_t, mu + 5*sigma))

### 3. Temporal Self-Attention
The core alpha engine utilizes a standard dot-product attention formulation to compute sequence-to-sequence dependencies across the 60-minute rolling window, exposing the "glass box" reasoning for execution triggers:

* Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V

---

## Core Modules

### `KaggleDataPipeline` & `MarketAuditor`
Handles the immediate ingestion and cleaning of structural data (e.g., missing timestamps, duplicated rows). Converts UNIX timestamps and resamples to a uniform 1-minute forward-filled frequency.

### `ProductionFeatureEngine`
Generates the core matrix of execution indicators. Emphasizes deployment speed by utilizing the `ta` library for robust, vectorized computations.
* **Trend:** EMA (15, 60), MACD (12, 26, 9)
* **Momentum:** RSI (14), ROC (15)
* **Statistical:** Z-Score (60-period rolling)

### `GlassBoxTransformer`
A PyTorch sequence classification model exposing its internal layer attentions.
* **Dimensionality:** 64 Model Dimensions (`d_model`)
* **Attention:** 4 Attention Heads
* **Depth:** 2 Transformer Blocks
* **Output:** Binary Logits + Extracted Temporal Heatmaps

---

## Baseline Alpha Evaluation

Before deploying the deep sequence model, a tree-based gradient boosting baseline (XGBoost) establishes the lower bound for predictive validity on the chronological validation split.

| Metric | Score (Baseline) |
| :--- | :--- |
| **Accuracy** | 0.5159 |
| **Precision** | 0.5189 |
| **Recall** | 0.4005 |
| **F1 Score** | 0.4521 |
| **ROC-AUC** | 0.5231 |

---

## Execution Dynamics & Explainability

The pipeline includes a dedicated `GlassBoxVisualizer` class. This module extracts the Q and K interaction matrices from the final Transformer layer and maps them directly over the target sequence price dynamics.

By outputting the self-attention heatmap against the 60-minute query step, quantitative researchers can audit exactly which historical micro-movements the agent is prioritizing before executing a simulated trade.
