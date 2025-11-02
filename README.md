#  Reinforcement-Driven Active Learning Framework for Real-Time Fraud Detection

###  Project Overview
This project presents an **adaptive Machine Learning framework** designed for **real-time fraud detection** in FinTech systems.  
Unlike traditional models, this approach continuously learns from **live streaming transactions** and adapts to new fraud patterns automatically using **reinforcement learning**, **active learning**, and **generative oversampling**.

---

##  Key Features

- **Reinforcement-Driven Learning:**  
  Uses the **LinUCB Bandit Algorithm** to dynamically select the best-performing model for each transaction.

- **Active Learning Integration:**  
  Reduces manual labeling by querying only the uncertain transactions, improving efficiency.

- **Generative Oversampling:**  
  Balances the dataset dynamically by generating realistic synthetic fraud samples (avoiding redundancy seen in SMOTE).

- **Online Learning (Streaming):**  
  The model updates itself after every transaction, adapting to **concept drift** in real time.

- **Real-Time Fraud Alerts:**  
  Enables immediate fraud detection and model updates with minimal delay.

---

##  Problem Statement

Financial institutions lose billions annually to undetected fraudulent transactions.  
Traditional ML models often fail to adapt to **evolving fraud patterns** (concept drift) and suffer from **severe class imbalance**, leading to **high false negatives**.

This project solves that by building an **adaptive, real-time learning system** that minimizes labeling effort and maintains high performance even as transaction behavior changes.

---

##  System Architecture

1. **Data Streaming Loop**  
   - Reads transactions one by one (simulating real-time input).  
   - Passes each transaction through the LinUCB Bandit model for prediction.

2. **Model Selector (LinUCB Bandit)**  
   - Chooses the best-performing model dynamically for each transaction.  
   - Updates its reward based on feedback (correct/incorrect prediction).

3. **Generative Oversampling**  
   - Generates new fraud samples to maintain balance between classes.

4. **Active Learning Module**  
   - Selectively queries only uncertain data points for manual labeling.

5. **Performance Monitor**  
   - Continuously tracks Accuracy, Precision, Recall, and F1-Score trends.  
   - Detects performance drops due to data drift and triggers adaptation.

---

##  Mathematical Overview

The framework is based on **Reinforcement Learning with Upper Confidence Bound (UCB)** logic:

![equation](https://latex.codecogs.com/svg.image?\color{White}a_t%20=%20\arg\max_a%20\left(%20\hat{\mu}_a%20+%20\alpha%20\sqrt{\frac{2\ln%20t}{n_a}}%20\right))



where:

- ![equation](https://latex.codecogs.com/svg.image?\bg_black\color{White}a_t) = chosen model (arm) at time *t*  
- ![equation](https://latex.codecogs.com/svg.image?\bg_black\color{White}\hat{\mu}_a) = estimated reward (performance) of arm *a*  
- ![equation](https://latex.codecogs.com/svg.image?\bg_black\color{White}n_a) = number of times arm *a* has been selected  
- ![equation](https://latex.codecogs.com/svg.image?\bg_black\color{White}\alpha) = exploration–exploitation constant

---

##  Performance Highlights

| Metric | Trend | Meaning |
|--------|--------|---------|
| **Accuracy** | Stays high (~1.0) | Model predicts non-fraud cases well |
| **Precision** | Slightly drops | Indicates concept drift in fraud ratio |
| **Recall** | ~0.75–0.8 | Continues detecting most frauds |
| **F1-Score** | Decreases mildly | Overall adaptability remains good |
| **MSE Trend** | Stabilizes after drift | Model re-learns and adapts to new data |

---

##  Tech Stack

- **Python 3.x**
- **River** (for streaming ML and online learning)
- **Pandas**, **NumPy**, **Matplotlib**
- **Scikit-learn**
- **Jupyter / Google Colab**

---

## Reference
- https://ieeexplore.ieee.org/document/9204849 
- https://ieeexplore.ieee.org/document/11010591

---

##  How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/<your-username>/Fraud-Detection-Framework.git
   cd Fraud-Detection-Framework
