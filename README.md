# Hotel Booking Cancellation Predictor

A Streamlit app that predicts whether a hotel booking will be **canceled**, using a single saved `scikit-learn` `Pipeline` (preprocessing + XGBoost model) trained in `Hotel_Booking_Cancellation_Prediction_refined.ipynb`.

- **Model:** XGBoost (selected as the best of 4 candidates: Logistic Regression, Decision Tree, Random Forest, XGBoost)
- **Test F1:** 0.755
- **Test ROC AUC:** 0.925
- **Decision threshold:** 0.54 (tuned on leakage-safe out-of-fold predictions, never on the test set)

## Repository contents

```
.
├── app.py                     # Streamlit app — loads the pipeline, no fitting happens here
├── final_pipeline.joblib      # Saved Pipeline: ColumnTransformer (imputers, FrequencyEncoder,
│                               # OneHotEncoder) + trained XGBoost model, fit once in the notebook
├── deployment_config.json     # Model name, tuned decision threshold, test metrics, expected feature columns
├── requirements.txt
├── .streamlit/config.toml     # App theme / server settings
├── .gitignore
└── Hotel_Booking_Cancellation_Prediction_refined.ipynb   # Full training notebook, for reference
```

## Data leakage audit (done before writing any deployment code)

This was the top priority for this deployment, so here's exactly what was verified in the notebook and enforced in `app.py`:

1. **Train/test split happens before any statistical fitting.** The notebook drops duplicate bookings and the two outcome-only columns (`reservation_status`, `reservation_status_date`) *before* splitting, then splits into `X_train`/`X_test`, and only *after that* builds `ColumnTransformer`/`Pipeline` objects. Every `.fit()` call in the notebook (imputers, the custom `FrequencyEncoder`, `OneHotEncoder`, `GridSearchCV`, the final model) is called on `X_train` only.
2. **The custom `FrequencyEncoder` is leakage-safe by construction.** It's a proper `BaseEstimator`/`TransformerMixin`: `fit()` learns value→frequency maps only from whatever rows it's given (a CV training fold, or `X_train` for the final fit), and `transform()` purely applies that already-learned mapping, mapping any unseen category to `0.0`. It's never re-fit on validation/test/user-input data.
3. **The decision threshold was tuned without touching the test set.** It comes from `cross_val_predict` out-of-fold probabilities on the training data; the test set is used exactly once, at the end, to report the final metrics saved in `deployment_config.json`.
4. **The saved artifact is the single object that was actually trained**, `final_model` (`.best_estimator_` from `GridSearchCV` for the winning model), dumped directly to `final_pipeline.joblib` right after evaluation. It is not rebuilt or re-fit for deployment.
5. **`app.py` performs zero fitting.** It calls `joblib.load()` once (cached with `st.cache_resource`) and only ever calls `pipeline.predict_proba()`. The one function it reimplements, `engineer_features()`, is copied verbatim from the notebook and is **purely deterministic arithmetic/lookups** (nights = weekend nights + week nights, a fixed month-name→number dictionary, log1p, etc.) — it learns nothing from data, so recomputing it at inference time on a single user-submitted row cannot leak anything. All *learned* preprocessing (imputation medians, frequency-encoding maps, one-hot categories) lives inside `final_pipeline.joblib`, not in `app.py`.
6. **Column order/set is enforced from the training artifact.** `app.py` reindexes the engineered input to `deployment_config.json["feature_columns"]` (the literal `list(X_train.columns)` saved at training time), so the `ColumnTransformer` always receives columns in the exact shape it was fit on.

**Conclusion:** the deployment reuses the exact fitted pipeline object from training, never refits any preprocessing step, and reproduces only deterministic feature engineering at inference time. No data leakage is introduced.

## Running locally

```bash
# 1. Clone the repo and enter it
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud

1. Push this folder to a **public** (or Community-Cloud-connected private) GitHub repository, keeping `app.py`, `final_pipeline.joblib`, `deployment_config.json`, and `requirements.txt` at the repo root (or note the subfolder path in step 4).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, then select this repository and branch.
4. Set **Main file path** to `app.py` (adjust if it's in a subfolder).
5. Click **Deploy**. Streamlit Cloud will install everything from `requirements.txt` and launch the app.
6. Any time you push new commits to the connected branch, the deployed app redeploys automatically.

**Note on `final_pipeline.joblib` size:** GitHub blocks files over 100 MB via normal `git push`. This file is a few MB, well under that limit, so a normal commit is fine — no Git LFS needed.

## How predictions are produced

1. You fill in the raw booking details in the form (the same raw fields the original dataset has, before feature engineering).
2. `engineer_features()` recomputes the same derived columns as the notebook (`total_nights`, `is_family`, `room_changed`, `prior_cancel_rate`, `lead_time_log`, etc.) — pure, deterministic transformations of the inputs you just gave.
3. The row is reindexed to match `feature_columns` from `deployment_config.json`.
4. `final_pipeline.joblib` (loaded once, cached) runs the row through its internal `ColumnTransformer` (`.transform()` only) and the trained XGBoost model's `.predict_proba()`.
5. The predicted cancellation probability is compared against the tuned decision threshold (0.54) to produce the final "Canceled" / "Honored" label, and both are shown in the UI.
