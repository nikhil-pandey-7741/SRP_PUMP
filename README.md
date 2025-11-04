# Multi-Label Fault Diagnosis of Sucker Rod Pump (SRP) Indicator Diagrams

This repository contains an experimental implementation of a **multi-label classification approach** for fault diagnosis in **Sucker Rod Pump (SRP)** systems.  
The methodology is inspired by the research paper:

 "**Multi-label learning for fault diagnosis of pumping units with one positive label (SPM-FDPU)", Applied Soft Computing, 2025.**  


The goal of this project is to detect **multiple co-occurring faults** from *indicator diagrams* even when only **one positive label** is available for training.

---

## 🔍 Problem Overview

In real-world SRP systems, **multiple faults can occur simultaneously**, such as:

| Fault Type | Abbreviation |
|----------|-------------|
| Insufficient Liquid Supply | ILS |
| Gas Impact | GI |
| Gas Block Effect | GBE |
| Pump Bumping (Top/Bottom) | TPB / BPB |
| Valve Leakage (Delivery / Suction) | DVL / SVL |
| Plunger Out of Barrel | POB |
| Sand Effect | SAND |
| Inertance | IN |

Traditional training data typically includes **only one labeled fault** per diagram, even though multiple may be present.  
This repository explores a model that can **predict multi-fault labels** from **single-positive supervision**.

---

## 🧪 Data Description

We currently use **synthetically generated indicator diagrams** produced from the **TAM** simulation software.

| Data Source | Type | Usage |
|------------|------|--------|
| TAM Simulation Outputs | Synthetic | Model Training |
| Real Field Indicator Diagrams | Real | Model Evaluation / Validation |

> ⚠️ Real field-labeled multi-fault training data is not available.  
> Therefore, domain generalization and robustness are key research challenges.

---

## 🏗 Model Overview

The model pipeline consists of:

1. **Indicator Diagram Preprocessing**  
   - Convert displacement–load curves to grayscale images
   - Normalize size and aspect ratio

2. **Feature Extraction (CSCID Inspired)**  
   - **Common Features:** HU Invariant Moments (7-dimensional)
   - **Specific Features:** CNN-based Sparse Convolution Feature Network (SCMF)

3. **Multi-Label Classification**
   - Trains using *single-positive labels*
   - Predicts *multi-label fault states* at inference




