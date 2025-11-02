# Real-Time Credit Card Fraud Detection using LinUCB (Reinforcement Learning)

This project implements a **real-time streaming fraud detection system** using a combination of **traditional machine learning models** and a **Reinforcement Learning agent (LinUCB)** to dynamically select the best model for each incoming transaction.

---

## Overview

The system processes credit card transaction data in a streaming fashion, where each transaction arrives one at a time.  
It uses **contextual bandit reinforcement learning (LinUCB)** to decide which model (arm) performs best for each transaction based on its context (features).

The pipeline combines **online learning** and **reinforcement learning**, making it suitable for real-world fraud detection scenarios where data arrives continuously.

---

## Workflow Steps

### 1. Setup and Imports
- Loads all required libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, and custom modules.
- Configures visualization settings for better plots.

### 2. Load and Inspect Data
- Reads the dataset `creditcard.csv`.
- Sorts the data by transaction time and resets the index.
- Displays:
  - Dataset shape  
  - Column names  
  - Fraud ratio (percentage of fraudulent transactions)  
  - Minimum and maximum transaction times

### 3. Feature Selection and Scaling
- Selects feature columns (excluding `Time` and `Class`).
- Converts features and labels into NumPy arrays for efficient computation.
- Uses **StandardScaler** for normalization, fitted on the first 5,000 samples (warmup block).

### 4. Streaming Data Generator
- Implements a Python generator function to yield data in small batches, simulating **real-time data streaming**.

### 5. LinUCB Agent (Reinforcement Learning)
- Implements the **LinUCB** algorithm (contextual bandit):
  - Selects the best model (arm) for each transaction.
  - Receives feedback (reward) based on correctness.
  - Updates parameters to improve future decisions.

### 6. Initialize Models
- Sets up **three machine learning models** as arms:
  1. Hoeffding Tree Classifier  
  2. Gaussian Naive Bayes  
  3. Logistic Regression  
- Initializes the LinUCB agent with these models.

### 7. Warmup Phase
- Trains all models on the first 5,000 transactions to provide an initial understanding of data.
- Helps models avoid the “cold start” problem.

### 8. Streaming Loop
Processes the remaining transactions **sample by sample**:
- Scales the current feature.
- Extracts context (subset of features).
- LinUCB selects the most suitable model.
- The model predicts the fraud probability.
- Calculates a reward (1 for correct, 0 for wrong).
- Updates the LinUCB agent and the chosen model.
- Every 20,000 samples → Evaluates:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix

### 9. Final Evaluation
At the end of streaming:
- Computes overall metrics:
  - Accuracy, Precision, Recall, F1-score, MSE
- Displays confusion matrix and summary.

### 10. Metric Trends and Preview
- Plots metrics (Accuracy, Precision, Recall, F1, MSE) over time.
- Shows first 2,000 predictions:
  - True label
  - Predicted label
  - Fraud probability

### 11. Fraud vs Non-Fraud Analysis
- Visual comparisons of fraudulent vs normal transactions:
  - Transaction amount distribution
  - Model confidence levels
  - Box plots and scatter plots
  - Cumulative detections over time
  - Correlation heatmaps
  - Precision-recall tradeoffs at various thresholds

### 12. Summary Statistics
Summarizes:
- Total predictions made
- Count of fraud vs normal detections
- Correct vs incorrect predictions
- Overall model performance

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python 3.x |
| ML Models | Logistic Regression, GaussianNB, Hoeffding Tree |
| RL Algorithm | LinUCB (Contextual Bandit) |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn Metrics |

---

## Dataset

Dataset: [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- **Rows:** 284,807  
- **Fraud Cases:** 492 (~0.17%)  
- **Features:** 30 (V1–V28 are PCA-transformed features, plus `Amount` and `Time`)

---

## Key Concepts

| Concept | Description |
|----------|--------------|
| **Streaming ML** | Continuously updates model as new data arrives. |
| **LinUCB** | Reinforcement Learning algorithm for online model selection. |
| **Warmup Block** | Initial samples used to train models before RL starts. |
| **Reward Function** | Binary reward: 1 (correct) or 0 (incorrect). |
| **Contextual Bandit** | Chooses the best model (arm) using partial feedback. |

---

## Running the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fraud-detection-linucb.git
cd fraud-detection-linucb
