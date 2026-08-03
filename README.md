# Bank Marketing Dataset — Machine Learning Project

## Task

The **Bank Marketing** dataset contains information from direct marketing campaigns (telephone calls) conducted by a banking institution. The goal is to build a **binary classification** model that predicts whether a client will subscribe to a term deposit (target variable `y`). The dataset was obtained from **Kaggle** and contains **41,176 records** and **21 variables**.

The original dataset is highly imbalanced: approximately **89%** of clients **did not** subscribe to a term deposit, while only **11%** did.

---

## Approach

### Evaluation Metric

Due to the class imbalance, **accuracy** is not a very informative metric. Therefore, **F1-score** was chosen as the primary evaluation metric, as it balances **precision** and **recall**. In addition, **ROC-AUC**, **precision**, and **recall** were also evaluated.

### Exploratory Data Analysis

During the EDA, several hypotheses about the influence of individual features were formulated and tested:

* **Age:** Clients younger than 25 and older than 60 were more likely to subscribe to a term deposit.
* **Occupation:** Students and `entrepreneur` clients had higher conversion rates, while `blue-collar` clients had lower conversion rates.
* **Marital status:** Single clients subscribed to a term deposit more often than married clients.
* **Education:** Having a university degree was positively associated with subscribing to a term deposit.
* **Month:** April, October, September, March, and December had the highest conversion rates, while May had the lowest.
* **Contact type:** Clients contacted via `cellular` converted more often.
* **Previous contact:** Having been contacted before the current campaign significantly increased the probability of subscribing to a term deposit.

### Preprocessing and Feature Engineering

| Action                                     | Details                                                                                                               |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Removed `duration`                         | Not available before the phone call, so it cannot be used in a real-world prediction scenario                         |
| Removed `default`                          | Almost all values are `no` or `unknown`, providing very little useful information                                     |
| `campaign` → log transformation + clipping | Log transformation to improve the distribution; values were clipped at 2.5, where the conversion rate dropped to zero |
| `pdays` → binary `pdays_was_contacted`     | The value 999 means "not contacted previously"; replaced with a binary 0/1 feature                                    |
| Created `had_previous_contact`             | Binary feature indicating whether the client had been contacted before the current campaign                           |
| Categorical variables → One-Hot Encoding   | Applied to all categorical variables                                                                                  |
| Numerical variables → StandardScaler       | Used for models sensitive to feature scaling (Logistic Regression and kNN)                                            |
| `scale_pos_weight` for XGBoost             | Approximately 7.88, accounting for the class imbalance                                                                |

A noticeable correlation was observed among the macroeconomic variables (`emp.var.rate`, `euribor3m`, and `nr.employed`). For Logistic Regression, models with and without the correlated variables were compared, both with regularization.

---

## Models

A total of **five models** were trained, including two hyperparameter tuning approaches for XGBoost:

1. **Logistic Regression** – a baseline linear model with regularization, tuned using `RandomizedSearchCV`.
2. **k-Nearest Neighbors (kNN)** – combined with PCA for dimensionality reduction and tuned using `RandomizedSearchCV`.
3. **Decision Tree** – tuned using `RandomizedSearchCV`.
4. **XGBoost (RandomizedSearch)** – tuned using Scikit-learn's `RandomizedSearchCV`.
5. **XGBoost (Hyperopt)** – tuned using Bayesian optimization with the `hyperopt` library.

For all models, the optimal **decision threshold** (instead of the default value of 0.5) was selected by searching thresholds from **0.10 to 0.90** with a step size of **0.01** on the validation set.

---

## Results

| Model                      | Dataset    | Threshold | Accuracy | Precision | Recall | F1     | ROC-AUC |
| -------------------------- | ---------- | --------- | -------- | --------- | ------ | ------ | ------- |
| Logistic Regression        | train      | 0.65      | 0.8644   | 0.4238    | 0.5664 | 0.4848 | 0.7945  |
| Logistic Regression        | validation | 0.65      | 0.8701   | 0.4428    | 0.5927 | 0.5069 | 0.7996  |
| kNN                        | train      | 0.21      | 0.8707   | 0.4406    | 0.5462 | 0.4877 | 0.8242  |
| kNN                        | validation | 0.21      | 0.8765   | 0.4606    | 0.5603 | 0.5056 | 0.7889  |
| Decision Tree              | train      | 0.48      | 0.8703   | 0.4418    | 0.5750 | 0.4997 | 0.7905  |
| Decision Tree              | validation | 0.48      | 0.8763   | 0.4626    | 0.6056 | 0.5245 | 0.8008  |
| XGBoost (RandomizedSearch) | train      | 0.58      | 0.8743   | 0.4545    | 0.5794 | 0.5094 | 0.8161  |
| XGBoost (RandomizedSearch) | validation | 0.58      | 0.8804   | 0.4757    | 0.6002 | 0.5307 | 0.8133  |
| XGBoost (Hyperopt)         | train      | 0.69      | 0.8858   | 0.4942    | 0.5740 | 0.5311 | 0.8617  |
| XGBoost (Hyperopt)         | validation | 0.69      | 0.8854   | 0.4926    | 0.5744 | 0.5303 | 0.8118  |

> **Final model:** **XGBoost (RandomizedSearch)** — achieved the best balance between F1-score and ROC-AUC, with a slightly higher recall than the Hyperopt-tuned model.

---

## Feature Importance (XGBoost)

| Feature               | Importance | Comment                                                                                                                  |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| `nr.employed`         | 0.451      | The dominant feature. Fewer employed people → higher probability of subscribing to a term deposit (less stable economy). |
| `emp.var.rate`        | 0.091      | Employment variation rate: lower values → higher probability of subscription.                                            |
| `month_may`           | 0.079      | May had the lowest conversion rate, consistent with the EDA.                                                             |
| `euribor3m`           | 0.076      | Lower interest rates → more term deposit subscriptions.                                                                  |
| `month_oct`           | ~0.04      | October had a high conversion rate.                                                                                      |
| `pdays_was_contacted` | ~0.03      | Previous contact increases the probability of subscription.                                                              |
| `poutcome_success`    | ~0.03      | A successful previous campaign has a positive effect.                                                                    |

From a common-sense perspective, the ranking of feature importance appears reasonable.

---

## SHAP Analysis

The SHAP analysis of the top 10 features confirmed the conclusions drawn from the feature importance analysis:

* **`nr.employed`** shows a negative relationship with the target: higher employment levels are associated with a lower probability of subscribing to a term deposit.
* **`emp.var.rate`** and **`euribor3m`** exhibit a similar pattern: lower values increase the probability of subscription.
* **`cons.conf.idx`** is particularly interesting: lower consumer confidence in the economy paradoxically increases the probability of subscribing to a term deposit.
* **`pdays_was_contacted`** shows a clear positive effect: clients who had been contacted previously tend to have positive SHAP values.
* **`campaign`** shows the opposite pattern: the more times a client is contacted during the current campaign, the lower the predicted probability of subscribing.

---

## Error Analysis

### False Negatives (the model misses actual subscribers)

* Clients observed under relatively stable macroeconomic conditions are often underestimated by the model.
* The largest proportion of False Negatives occurs in **April** and **September**, despite these months having high conversion rates in the EDA.
* Underrepresented groups such as **`unemployed`**, **`student`**, and **`retired`** are more likely to be misclassified because the model has fewer examples from which to learn.
* The median predicted probability for False Negatives is **0.43**, well below the selected threshold of **0.69**.

### False Positives (the model overestimates non-subscribers)

* Clients with **`poutcome_success`** account for **23.7%** of False Positives, indicating that the model relies too heavily on this feature.
* The largest proportion of False Positives occurs in **December**, **October**, and **March**, months with relatively few observations.
* The median predicted probability for False Positives is **0.72**, indicating that the model is often quite confident in its incorrect predictions.

---

## Ideas for Improvement

1. **Lower the decision threshold** from **0.69** to approximately **0.45–0.50** to identify more potential subscribers.
2. **Add an interaction feature** between **`nr.employed`** and **`pdays_was_contacted`** so that the model can better distinguish clients under different macroeconomic conditions.
3. **Apply oversampling** to underrepresented groups (`retired`, `student`, `unemployed`) and months with few observations (December, October, and March) to improve model stability.
