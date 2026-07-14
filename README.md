## CA-Property_Price_Modeling_IDXExchange-- Daniel Cai

This repository contains my individual work for the IDX Exchange Data Science Internship. The goal is to build a ML workflow to predict the ClosePrice of single-family properties in CA using historical CRMLS sold property data. This project includes data exploration, preprocessing, feature engineering, model training, evaluation, and results.

### Week 1 — Orientation & Setup
- Established FTP connection via FileZilla to IDX Exchange server
- Downloaded minimum 6 months of CRMLSSold filled CSV files from `/raw/California`
- Reviewed Trestle Property MetaData PDF to understand all field definitions
- Documented key column definitions and feature utility assessment

**Deliverable:** Feature utility assessment (`IDX_Week1_Feature_Assessment.pdf`)

---

### Week 2 — Data Exploration
- Loaded and merged 6+ months of CRMLSSold data into a single pandas DataFrame (~397K rows)
- Filtered to Residential / SingleFamilyResidence only
- Explored distributions of key features: `ClosePrice`, `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, `LotSizeSquareFeet`
- Identified and documented outliers (e.g. LotSize up to 2B sqft, BathroomsTotalInteger up to 3000)
- Analyzed feature relationships via scatter plots and correlation heatmap
- Built amenity density heatmap across boolean and count features

**Deliverable:** `Jupyter notebook 01_exploration.ipynb`

---

### Week 3 — Data Preprocessing
- Dropped columns with >80% missing data and low predictive signal
- Imputed remaining missing values based on domain knowledge (missing YN = 0, missing HOA = 0)
- Removed data leakage risk by dropping `ListPrice` (0.97 correlation with `ClosePrice`)
- Extracted time-based features from `CloseDate` (month, quarter, year)
- Created temporal train/test split — most recent month as test set, preceding months as training set

**Deliverable:** `Jupyter notebook 02_Pre_Process.ipynb`

### Week 4 - Basic Linear Regression Model

- Loaded pre-split train/test data and separated features from target (`ClosePrice`)
- Dropped ID and date columns plus leakage-prone fields
- Encoded boolean-like YN columns to 0/1 and label-encoded remaining categorical columns
- Removed rows with missing target values from train and test sets
- Imputed missing feature values with median strategy (fit on training set only)
- Standardized features with `StandardScaler` (fit on training set only)
- Trained baseline `LinearRegression` model
- Evaluated on held-out test set: R², RMSE, and median APE
- Visualized predicted vs. actual prices with identity reference line
- Serialized trained model and scaler with `joblib` for reuse

**Deliverable:** `Jupyter notebook 03_baseline_model.ipynb`
