# Predicting 30-Day Hospital Readmission in Diabetic Patients

Machine Learning module project: traditional ML (scikit-learn) vs deep learning (TensorFlow/Keras) on the
**Diabetes 130-US Hospitals (1999-2008)** dataset from the UCI Machine Learning Repository
(101,766 encounters, 50 features). Task: binary classification of readmission within 30 days.

## Links
- Report (PDF): <add link>
- Google Drive (raw data zip + saved models): <add link, "Anyone with the link can view">
- Video presentation: <add link>

## Repository contents
| File | Purpose |
|---|---|
| `diabetes_readmission.ipynb` | Full pipeline: EDA, cleaning, feature engineering, ML, DL, evaluation |
| `requirements.txt` | Python dependencies |

## How to reproduce
1. `pip install -r requirements.txt` (or open the notebook in Google Colab).
2. Run the notebook top to bottom. It downloads the dataset from UCI automatically on first run
   (or place `diabetic_data.csv` next to the notebook; it is also in the Drive folder).
3. Trained models are written to `artifacts/` and are available in the Drive folder.

## Method summary
- Cleaning: remove expired/hospice discharges, keep one encounter per patient (avoids leakage), drop `weight`.
- Feature engineering: ICD-9 grouping, prior-visit totals, medication-change counts, per-day intensity ratios.
- ML: Logistic Regression, Linear/RBF SVM, Random Forest, Histogram Gradient Boosting, randomized search CV.
- DL: Sequential and Functional (residual MLP) Keras models, `tf.data` pipeline, class weights, early stopping.
- Evaluation: ROC-AUC, PR-AUC, precision/recall, F2 threshold tuning, learning curves, confusion matrices, error analysis.

## Results
Held-out test set (13,998 encounters; positive rate 9.0%). Thresholds maximise F2 and were chosen without using the test labels.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F2 |
|---|---:|---:|---:|---:|---:|
| Random forest (tuned) | 0.656 | 0.179 | 0.124 | 0.679 | 0.358 |
| Residual MLP (best DL) | 0.652 | 0.165 | 0.108 | 0.831 | 0.356 |
| Histogram gradient boosting (tuned) | 0.650 | 0.183 | 0.117 | 0.700 | 0.350 |
| Logistic regression (tuned) | 0.649 | 0.169 | 0.114 | 0.762 | 0.357 |

The random-forest 95% bootstrap interval for ROC-AUC is 0.640–0.673. The neural-network interval is 0.636–0.670. They overlap. Feature engineering changed cross-validated ROC-AUC by only +0.002 to +0.006. Figures from the executed notebook are in `figures/`. The written report is `Predicting 30-Day Hospital Readmission in Diabetic Patients - Report.docx`.

## Note
The raw dataset is intentionally not committed to this repository.
