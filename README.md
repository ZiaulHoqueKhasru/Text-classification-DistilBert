# 🤖 Text-Classification with DistilBERT: From Math to Production

[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?style=for-the-badge&logo=amazonsagemaker&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready MLOps pipeline that fine-tunes **DistilBERT** for **multi-class text classification** and deploys it on AWS using **SageMaker + Lambda + API Gateway**. This repo is built to bridge **Transformer fundamentals** (attention math, tokenization) with **real deployment patterns** (scalable inference, serverless API, reliability checks).

---

## 🎥 Codebase Walkthrough (Video)
Watch a complete end-to-end walkthrough of the codebase and deployment flow:  
👉 **Demo Video (Google Drive)**: [Open the walkthrough](https://drive.google.com/file/d/17tNvxM6ephlDrriklnt7SKEDHlp2vB1N/view?usp=drive_link)

---

## 🧠 Theory Notes (Attention + Tokenization + Proofs)
For a deep dive into the **Attention Mechanism**, **tokenization algorithms**, and supporting math notes (with screenshots):  
👉 **Notion Deep-Dive**: [Explore the Transformer Architecture](https://www.notion.so/Transformer-Architecture-2cf1bf3586bc800a8da6de9e6534a302?source=copy_link)

---

## ✅ What This Project Covers
- **Data ingestion** from the UCI text dataset (news-style classification)
- **S3-based dataset management** (raw + processed artifacts)
- **EDA** (class balance, length distribution, truncation strategy)
- **GPU fine-tuning** with SageMaker Training Jobs
- **Real-time hosting** via SageMaker Endpoint
- **Serverless API** using Lambda + API Gateway
- **Testing** for correctness (unseen samples) + latency benchmarking

---

## 🏗️ Architecture

flowchart LR
  %% =========================
  %% Architecture Diagram
  %% =========================

  A["Raw Dataset (UCI)"] --> B["S3: Raw Bucket"]
  B --> C["Preprocessing + EDA"]
  C --> D["S3: Processed Data"]
  D --> E["SageMaker Training Job (GPU)"]
  E --> F["S3: Model Artifacts"]
  F --> G["SageMaker Model"]
  G --> H["SageMaker Endpoint (Real-time)"]

  I["Client / App"] --> J["API Gateway (REST)"]
  J --> K["AWS Lambda (Inference Bridge)"]
  K --> H
  H --> K
  K --> J
  J --> I
