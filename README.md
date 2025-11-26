# End-to-End Computer Vision Pipeline on AWS SageMaker ☁️📷

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20S3-orange?style=for-the-badge&logo=amazon-aws)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red?style=for-the-badge&logo=pytorch)
![Status](https://img.shields.io/badge/Status-Work%20In%20Progress-yellow?style=for-the-badge)
## 📋 Project Overview

This project demonstrates a complete **Machine Learning Operations (MLOps)** workflow for Computer Vision using **Amazon SageMaker**. 

Instead of training models locally, this pipeline orchestrates cloud resources to train a **ResNet18** model via Transfer Learning on the **Hymenoptera Dataset** (Ants vs. Bees). The project utilizes SageMaker's **Script Mode**, decoupling the training logic from the infrastructure code, simulating a real-world production environment.

### Key Features
* **Infrastructure as Code:** Automatic S3 bucket provisioning and data upload via Boto3.
* **Scalable Training:** Usage of AWS SageMaker Estimators to launch ephemeral EC2 instances for training.
* **Script Mode:** Custom PyTorch training script (`train.py`) injected into AWS managed containers.
* **Cloud Inference:** Model deployment to a real-time endpoint for predictions.

---

## 🏗️ Architecture

The pipeline follows the standard AWS ML lifecycle:

1.  **Data Ingestion:** Raw images are uploaded from local environment to **Amazon S3**.
2.  **Training:** SageMaker launches a managed instance (e.g., `ml.m5.large`), downloads data from S3, executes the training script, and uploads model artifacts (`model.tar.gz`) back to S3.
3.  **Deployment:** The trained artifact is deployed to an endpoint for real-time inference.

---

## 📂 Project Structure

```text
aws-sagemaker-cv-pipeline/
│
├── sagemaker_entry_point/      # Code that runs INSIDE the AWS Container
│   ├── train.py                # PyTorch training logic (ResNet18)
│   └── requirements.txt        # Dependencies for the training container
│
├── notebooks/                  # Orchestration Notebooks (The "Remote Control")
│   ├── hymenoptera/            # Main project notebooks
│   │   ├── 01_setup_data.ipynb # Data Engineering: S3 creation & Upload
│   │   ├── 02_training.ipynb   # Model Training: SageMaker Estimator
│   │   └── 03_inference.ipynb  # Deployment & Prediction
│   │   ├── ppe /                  # Personal Protective Equipment project notebooks
│   │   │   ├── ...                  # (Similar structure as hymenoptera)
│   └── utils/                  # Helper functions
│
├── data/                       # Local raw data (Ignored by Git)
├── requirements.txt            # Local dependencies (boto3, sagemaker)
└── README.md                   # Project documentation



