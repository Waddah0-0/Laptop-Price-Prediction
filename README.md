## Laptop Price Prediction – Data Science Workflow

This project predicts **laptop prices** using two regression models (Random Forest and XGBoost) built on a real‑world laptop dataset and deployed with a simple **Streamlit GUI**.

The work follows a typical **data science workflow**:

- **Data collection** (web scraping)
- **Data inspection and cleaning**
- **Exploratory Data Analysis (EDA)**
- **Model building and evaluation**
- **Saving models**
- **Building a GUI for predictions**

---

## 1. Project Structure

- `Code/`
  - `Scrape.ipynb`: scraping / collecting the raw laptop data
  - `Inspect&Clean.ipynb`: inspecting, cleaning, and transforming the raw data
  - `EDA.ipynb`: exploratory data analysis and visualizations
  - `Model.ipynb`: model training, evaluation, and saving (`Random Forest` and `XGBoost`)
  - `GUI.py`: Streamlit web app to use the trained models
- `Dataset/`
  - `laptops_Dataset.csv`: original/raw dataset
  - `Cleaned_data.csv`: cleaned version of the most important columns
  - `data_without_outliers.csv`: final numeric dataset used for modeling
- `Models/`
  - `Random_Forest_Model.joblib`: trained Random Forest regressor
  - `XGBoost_Model.joblib`: trained XGBoost regressor

---

## 2. Data Collection (Scrape.ipynb)

In `Code/Scrape.ipynb` the goal is to **collect laptop specifications and prices** from an online source.

Typical steps:

- Send HTTP requests to the website pages that list laptops.
- Parse the HTML using a library like `BeautifulSoup`.
- Extract fields such as:
  - Brand and product name
  - Processor
  - RAM
  - Storage (HDD/SSD)
  - GPU (Video graphics)
  - Display size and resolution
  - Weight
  - Price
- Store the scraped records into a **DataFrame**.
- Save the raw data as `Dataset/laptops_Dataset.csv` for further processing.

Result: a raw dataset with many text columns, missing values, and inconsistent formats.

---

## 3. Data Inspection and Cleaning (Inspect&Clean.ipynb)

In `Code/Inspect&Clean.ipynb` the focus is to **understand the dataset and make it usable for modeling**.

Main steps:

1. **Initial inspection**
   - Load `Dataset/laptops_Dataset.csv`.
   - Check shape, column names, data types, and missing values (`df.info()`, `df.isnull().sum()`).
2. **Selecting relevant columns**
   - Choose the columns that most affect price, for example:
     - `Processor`, `RAM`, `Hard drive`, `Display`, `Video graphics`,
     - `Display Resolution`, `Display Refresh Rate`, `Weight`, `Price`.
   - Create a subset `VIC` with only these columns.
3. **Handling missing values**
   - Fill missing `Display`, `Display Resolution`, `Display Refresh Rate`, `Weight`, `Video graphics` with:
     - either the **mode** (most frequent value)
     - or specific default values (e.g. `"15.6"`, `"1920 x 1080"`, `"144 Hz"`, `"2.29 kg"`).
4. **Removing duplicates**
   - Check for duplicated rows and drop them to create `VIC_df`.
5. **Feature extraction from text**
   - Convert string columns into numeric features using regex:
     - `RAM`: extract integer GB.
     - `Weight`: extract numeric kg.
     - `Screen size`: extract inches from the `Display` string.
     - `Display Refresh Rate`: extract Hz as integer.
     - `GPU`: extract a numeric identifier from `Video graphics`.
     - `Storage`: extract capacity and convert:
       - `1 TB → 1000 GB`, `2 TB → 2000 GB`.
     - `CPU`: extract a 4+ digit code from the `Processor` (e.g., 13420, 14900).
   - Split `Display Resolution` into:
     - `Screen Width` and `Screen Height` (e.g., 1920 and 1080).
6. **Type conversion and missing numeric values**
   - Convert extracted columns to numeric types (`int`/`float`).
   - Fill missing numeric values like `GPU`, `CPU`, `Storage` with the **median**.
7. **Final cleaned dataset**
   - Reorder and keep the final set of features in `VIC_Cleaned`, for example:
     - `Processor`, `Display`, `Video graphics`, `Hard drive`, `Storage`, `RAM`,
       `Display Resolution`, `Screen size`, `Screen Height`, `Screen Width`,
       `Display Refresh Rate`, `CPU`, `GPU`, `Weight`, `Price`.
   - Save as:
     - `Dataset/Cleaned_data.csv`
     - and later as `Dataset/data_without_outliers.csv` after further processing.

Result: a **clean numeric dataset** where important laptop specs are turned into usable numeric features, ready for modeling.

---

## 4. Exploratory Data Analysis (EDA.ipynb)

In `Code/EDA.ipynb` the goal is to **understand relationships in the cleaned data** and check if it is suitable for modeling.

Typical EDA steps:

- Load `Dataset/Cleaned_data.csv` or `Dataset/data_without_outliers.csv`.
- Look at summary statistics for each feature (mean, std, min, max).
- Plot distributions of:
  - `Price`
  - `RAM`, `Storage`, `Screen size`, `Weight`, etc.
- Check correlations between numeric features and `Price`.
- Plot relationships such as:
  - Price vs. RAM
  - Price vs. Storage
  - Price vs. Screen size or Resolution
- Identify and possibly **remove outliers**, saving the resulting dataset as:
  - `Dataset/data_without_outliers.csv`.

Result: better understanding of which features are most associated with laptop price and a cleaned dataset without extreme outliers.

---

## 5. Model Building and Evaluation (Model.ipynb)

In `Code/Model.ipynb` two regression models are trained on the processed data.

### 5.1 Data preparation

- Load `Dataset/data_without_outliers.csv` as `df`.
- Drop all object (string) columns:
  - `obj_cols = df.select_dtypes(include=["object"]).columns`
  - `data = df.drop(columns=obj_cols)`
- Define:
  - `y = data["Price"]`
  - `X = data.drop(columns=["Price"])`
- Split into train and test sets:
  - `train_test_split(X, y, test_size=0.2, random_state=10)`

### 5.2 Model 1 – Random Forest Regressor

- Initialize:
  - `RandomForestRegressor(n_estimators=100, random_state=0)`
- Fit on training data and predict on test data.
- Evaluate with:
  - **Mean Absolute Error (MAE)**
  - **R² score**
  - **MAPE (mean absolute percentage error)**
- Visualize:
  - Actual vs. predicted prices (scatter plot).
  - Distribution of prediction errors (histogram).
- Save the trained model as:
  - `Models/Random_Forest_Model.joblib` using `joblib.dump`.

### 5.3 Model 2 – XGBoost Regressor

- Initialize:
  - `XGBRegressor(n_estimators=250, learning_rate=0.05, n_jobs=8)`
- Train on the same `X_train`, `y_train`.
- Evaluate with MAE, R², and MAPE.
- Save the model as:
  - `Models/XGBoost_Model.joblib`.

### 5.4 Model comparison

- Compute MAE and R² for both models.
- Plot bar charts comparing:
  - MAE (Random Forest vs XGBoost)
  - R² (Random Forest vs XGBoost)
- Conclusion in the notebook:
  - Both models have **similar performance** on this dataset, so either can be used in practice.

---

## 6. Streamlit GUI (GUI.py)

In `Code/GUI.py` a simple **web interface** is built using Streamlit so that a user can input laptop specs and get predicted prices from both models.

Main ideas:

- Load the same dataset used for modeling (`Dataset/data_without_outliers.csv`).
- Reconstruct the same **feature columns** used during training:
  - Drop object columns, keep numeric features, exclude `Price`.
- Load the saved models from:
  - `Models/Random_Forest_Model.joblib`
  - `Models/XGBoost_Model.joblib`
- Build a form with `st.number_input` for each numeric feature, using the mean value as a default.
- When the user clicks **Predict**:
  - Construct a one‑row `DataFrame` with the input features.
  - Call both models to get predictions.
  - Display the predicted prices for:
    - Random Forest
    - XGBoost

This makes the project **interactive** and easy to demonstrate.

---

## 7. How to Run the Project

### 7.1 Requirements

Install the main libraries (for example using `pip`):

```bash
pip install streamlit pandas scikit-learn xgboost joblib matplotlib
```

### 7.2 Running the notebooks

1. Open the project in Jupyter, VS Code, or other IDE.
2. Run the notebooks in roughly this order:
   1. `Code/Scrape.ipynb` (optional if data already saved)
   2. `Code/Inspect&Clean.ipynb`
   3. `Code/EDA.ipynb`
   4. `Code/Model.ipynb`
3. This will ensure:
   - Cleaned datasets are created in `Dataset/`.
   - Trained models are saved in `Models/`.

### 7.3 Running the Streamlit app

From the project root folder:

```bash
cd "E:\Waddah\Alex\VS\Metho\DS Methodology Final Project"
streamlit run Code/GUI.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## 8. Summary

This project shows a full **end‑to‑end data science pipeline** on a laptop pricing problem:

- Collect data from the web.
- Clean and transform messy text features into numeric variables.
- Explore the data and remove outliers.
- Train and compare two regression models (Random Forest and XGBoost).
- Save the models and expose them through an easy‑to‑use Streamlit interface.

It can be extended by adding more features, hyperparameter tuning, cross‑validation, or deployment to a cloud platform.


