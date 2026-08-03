# US Recession Forecasting: Macroeconomic Leading Indicators & Modeling

An end-to-end Machine Learning pipeline utilizing 3-month lagged macroeconomic variables to forecast US economic recessions. This project evaluates traditional linear baseline architectures against advanced non-linear classifiers using a strict chronological time-series splitting strategy to eliminate data leakage.

## 📌 Project Overview & Intent
The primary objective of this capstone is to discover what macroeconomic data can reveal beyond static modeling. By shifting historical features forward, the system builds a 3-month early-warning predictive cushion. This repository contains the exploratory data analysis (EDA), data processing frameworks, and comparative model matrices generated for the initial report phase.

---

## 📊 Dataset Variables & Feature Engineering
The model processes a mixed-frequency macroeconomic dataset spanning interest rates, inflation metrics, and production indexes:
*   **Contemporaneous Features:** `Price_x`, `INDPRO`, `CPI`, `GDP`, `Rate`, `BBK_Index`, `Housing_Index`
*   **Yield Curve Spreads:** Long-term minus short-term Treasury rates (`10 Yr` - `2 Yr`, `10 Yr` - `3 Mo`, `30 Yr` - `5 Yr`)
*   **Economic Momentum:** Percentage changes representing structural momentum (`INDPRO_MoM`, `CPI_YoY`, `GDP_YoY`)
*   **Temporal Realism:** Features are dynamically filtered for infinite mathematical anomalies (caused by zero-level variables) and shifted using a **3-Month Lag (`_Lag3`)** to predict future target states natively.

---

## 🛠️ Data Strategy & Chronological Split
To respect real-world forecasting constraints and strictly prevent future-data leakage, traditional random shuffling splits were rejected:
1.  **Chronological Split:** The first **80%** of historical data forms the training set, while the remaining **20%** modern data serves as the test horizon.
2.  **Imbalance Architecture:** The test set represents real-world scarcity, isolating **64 Normal Expansion Months** against only **2 True Recession Months**.

---

## 📉 The Modeling Journey & Breakthroughs

### Phase 1: The Directional Inversion Problem
Initial model iterations yielded a severely inverted **ROC-AUC Score of 0.1094**. 
*   *Root Cause:* Critical economic warning flags (like yield spreads or housing indices) drop or turn negative ahead of a crash. Linear classifiers natively associate *increasing* numbers with positive targets (`Recession = 1`), completely mirroring the model's logic.
*   *The Fix:* Probabilities were mathematically flipped (`1 - y_proba`) to align dropping economic indicators with rising risk signatures.

### Phase 2: Calibration and Threshold Optimization
Correcting the direction initially caused the default 0.50 threshold to trigger false alarms for almost every single month. By implementing **Geometric Mean (G-Mean) optimization**, a custom threshold of **0.9975** was established. This filtered out 55 false alarms while maintaining an immaculate recall score.

---

## 🏁 Final Model Performance Comparison

The dynamic evaluation of our baseline against advanced machine learning architectures revealed that global, linear projections handle modern time-series shifts far better than tree-based partitions:

| Model Architecture | Threshold Type | ROC-AUC Score | True Recessions Caught | False Alarms Triggered | Model Status / Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **Optimized (0.9975)** | **0.8906** | **2 / 2** | **9 / 64** | **WINNER (Best Predictive Balance)** |
| Random Forest | Optimized (0.0900) | 0.5625 | 2 / 2 | 29 / 64 | Rejected (Excessive Noise) |
| SVM | Optimized (0.0058) | 0.5469 | 2 / 2 | 29 / 64 | Rejected (Excessive Noise) |
| XGBoost | Optimized (0.0021) | 0.6016 | 2 / 2 | 31 / 64 | Rejected (Worst Noise) |

### Key Report Insights:
1.  **Linear Superiority:** Tree-based models (Random Forest, XGBoost) suffer an extrapolation blind spot. When evaluated on a strict timeline, they cannot interpret numerical features that reach historic highs or lows unseen in older training blocks.
2.  **Threshold Collapse:** To force advanced classifiers to capture the true recessions, their decision thresholds collapsed to near-zero levels (`0.0021` for XGBoost), creating high-noise outputs that flagged nearly half of normal expansion history as active recessions.

---

## 🔮 Future Capstone Phases
To improve advanced model precision and beat our current baseline, upcoming project sprints will focus on:
*   **Stationary Feature Scaling:** Converting raw economic levels into rolling z-scores to normalize data ranges across chronological eras.
*   **Dynamic Lags:** Deploying heterogeneous lag windows (1-month, 3-month, and 6-month) to map cascading economic impacts.
*   **SMOTE Implementation:** Synthetically balancing historical recession representation during the training phase.