# Fraud Identification and Defaulter Location Inference

This project has two components:

1. **Task 1 — Loan Default Prediction**: a supervised ML pipeline that predicts whether a bank customer will default on a loan, using their credit and repayment behavior.
2. **Task 2 — Defaulter Location Inference Agent**: a multi-agent LLM pipeline that infers a person's most likely home and last-known location from scattered structured data (bank records, LinkedIn, ATM/UPI activity), useful for locating defaulters when a loan goes bad.

---

## Task 1: Loan Default Prediction

### Problem
Given a customer's credit limits, repayment history, risk scores (CRIFF), KYC status, and 12 months of monthly transaction trends, predict `TARGET = 0` (non-defaulter) or `1` (defaulter).

### Data
`HACKATHON_TRAINING_DATA.CSV` — not included in this repo (hackathon-provided data). Place it in the project root before running the notebook.

### Pipeline

**1. Cleaning & feature engineering**
- Y/N flags → 0/1; text durations (`"2 yrs 3 mon"`) parsed to months
- Categorical fields (income band, loan category, product type) encoded to ordinal codes
- Custom risk features built from the 12 monthly columns:
  - `overspend_ratio` — how much of a customer's spend exceeded their monthly limit
  - `max_consec_overspend` — longest streak of consecutive overspending months
  - `outbal_slope` / `outbal_is_declining` — trend in outstanding balance over the year
  - `slope_MTD` / `is_debit_declining_MTD` — trend in monthly debit activity
- 72 raw monthly columns dropped after these summary features are extracted, to reduce dimensionality

**2. Train / validation / test split**
- 64% train / 16% validation / 20% test, stratified on `TARGET`
- The validation set exists specifically to tune the classification threshold **without touching the test set** — see "Evaluation methodology" below.

**3. Preprocessing**
- Median imputation → Yeo-Johnson power transform (handles skew) → Min-Max scaling
- All transformers fit on train only, applied to val/test

**4. Feature selection**
- A preliminary XGBoost model is trained and explained with SHAP; the top 30 features by mean absolute SHAP value are kept for the final model

**5. Final model**
- LightGBM with `class_weight='balanced'`, tuned via `RandomizedSearchCV` (F1-scored, cross-validated)
- Classification threshold chosen by maximizing F1 on the **validation** set, then applied once to the held-out test set

### A methodological note (why there's no SMOTE-Tomek here)
An earlier version of this pipeline used **SMOTE-Tomek** to rebalance the training data before fitting. Two problems were found and fixed:

- **Threshold leakage**: the original threshold was chosen by scanning `precision_recall_curve` on the test set itself, then reporting F1 on that same test set — an optimistic, non-reproducible number. This is fixed by tuning the threshold on a dedicated validation split instead.
- **Miscalibration from SMOTE-Tomek**: after fixing the leak, the SMOTE-Tomek-trained model's predicted probabilities were checked and found to be severely miscalibrated (median predicted probability of default was **0.999**, vs. an actual ~11% default rate). Synthetic oversampling had made the model overconfident. Replacing SMOTE-Tomek with `class_weight='balanced'` on the real (unresampled) training data produced a properly spread, well-calibrated probability distribution (median ≈ 0.11) at no cost to F1 — and removed a ~20–30 minute step from every run.

### Final results (honestly evaluated: threshold tuned on validation, scored once on test)

| Metric | Value |
|---|---|
| F1 (defaulter class) | 0.60 |
| Recall (defaulter class) | 0.65 |
| Precision (defaulter class) | 0.56 |
| Accuracy | 90.7% |
| Threshold | 0.66 |

The model was tuned to favor **recall over precision** — for a bank, missing an actual defaulter is typically costlier than flagging a good customer for extra review, so a recall-oriented operating point was chosen for the headline result. Accuracy is not the primary metric here: the test set is ~89% non-defaulters, so a naive "predict no default for everyone" baseline already scores ~89% while being useless — F1/recall on the minority class is what matters.

### Requirements
```
pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap joblib
```

---

## Task 2: Defaulter Location Inference Agent

### Problem
When a borrower defaults and needs to be located, their most reliable current address may not be what's on file. This pipeline combines multiple weak, partial location signals (bank records, digital footprint, transaction geography) into one best-guess location with a confidence rating.

### Pipeline

**1. LinkedIn enrichment (`scrape_linkedin_locations`)**
- Uses Selenium to log into LinkedIn and scrape the profile location field for each person's LinkedIn URL in the dataset
- Requires manual login within a 30-second window when the script runs
- Output merged into `Final Dataset with LinkedIn.csv`

**2. Four-agent LLM pipeline (LangGraph + Groq `llama3-70b-8192`)**

| Agent | Role | Inputs scored |
|---|---|---|
| **Agent 1 — Static Verifier** | Infers **home location** | Branch (+2), DL number (+2), Vehicle registration (+1.5), Address (+1.5), Phone prefix (+0.5) |
| **Agent 2 — Activity Verifier** | Infers **last known location** | ATM activity (+3), UPI activity (+2.5), LinkedIn (+1.5), Frequent/last recorded location (+1) |
| **Agent 3 — Cross Validator** | Reconciles Agent 1 & 2 | Penalizes state-level conflicts (−2 each), flags missing critical fields (DL + Branch both absent), flags 3+ distinct cities, checks state agreement between home and last-known |
| **Agent 4 — Final Scorer** | Produces the final call | Confidence tiers: High (score ≥ 7), Medium (4–7), Low (< 4); appends "Manual Review Needed" for low-confidence, missing-data, or highly conflicting cases; +0.5 bonus if home and last-known agree on state |

Agents run sequentially via a LangGraph `StateGraph` (Agent 1 → 2 → 3 → 4 → END), with each agent's raw output logged for auditability.

**3. Output**
- Per-person: `Final_Predicted_Location`, `Prediction_Confidence` (High/Medium/Low), `Last_Known_Location`, and a full `Agent_Reasoning_Log`
- Saved to `multi_agent_location_predictions.csv`
- A bar chart of predicted-location frequency by state is generated at the end of the run

### Requirements
```
pandas selenium webdriver-manager langchain-groq langgraph matplotlib
```

### Setup
1. Set `GROQ_API_KEY` in the notebook (currently a placeholder — do not commit a real key).
2. Provide `Dummy Dataset Final.csv` with the expected columns (see `location_fields` in the script) in the project root.
3. Run the LinkedIn scraping cell first (interactive login required), then the agent pipeline.

### Notes / limitations
- The LinkedIn scraper depends on LinkedIn's current page structure (CSS selector) and manual login; it will break silently (`location = None`) if LinkedIn changes its markup, and scraping LinkedIn profiles this way may violate LinkedIn's Terms of Service — review this before using outside a controlled/educational context.
- Agent scoring rules (point values, thresholds) are heuristic, not learned or validated against ground-truth outcomes — worth revisiting if this is deployed rather than prototyped.
- The API key is currently hardcoded as a placeholder in the notebook; use an environment variable instead for real use.

---

## Repository structure
```
.
├── Task_1_Loan_Default_Prediction.ipynb       # loan default classifier
├── Task_2_Location_Detection_Agent.ipynb      # multi-agent location inference
├── checkpoints/                                # cached intermediate results (gitignore this)
├── backend/ frontend/                          # prototype web interface
└── README.md
```

## Disclaimer
This project was built for a hackathon using synthetic/dummy data. It is a prototype, not a production-ready system — see the methodology notes above for known limitations before extending or deploying it.