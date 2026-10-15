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
<paste your final results table here>

## Note
The raw dataset is intentionally not committed to this repository.
