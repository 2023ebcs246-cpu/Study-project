import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load the Dataset
# ==========================================
file_path = 'Custom_Crops_yield_Historical_Dataset.csv'
print(f"Loading dataset from {file_path}...")

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} total records.")
except FileNotFoundError:
    print("Error: File not found. Please make sure the CSV file is in the same directory.")
    exit()

# --- SETTINGS: Specific Crop and District for this demo ---
target_crop = 'cotton'
target_district = 'jhunjhunu'

print(f"\nFiltering data to train specifically for '{target_crop.upper()}'...")
df = df[df['Crop'].str.lower() == target_crop.lower()]
df = df[df['Dist Name'].str.lower() == target_district.lower()]

features = ['N_req_kg_per_ha', 'P_req_kg_per_ha', 'K_req_kg_per_ha', 'pH', 'Rainfall_mm']
target = 'Yield_kg_per_ha'

# ==========================================
# 2. Data Analysis (Exploratory Data Analysis)
# ==========================================
print(f"Total {target_crop.upper()} records available: {len(df)}")

# Create and save a visual graph of the Yield
plt.figure(figsize=(8, 5))
plt.hist(df[target].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title(f'Yield Distribution for {target_crop.upper()}')
plt.xlabel('Yield (kg/ha)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('yield_distribution.png')
print("[!] A yield distribution chart has been saved as 'yield_distribution.png'.")

# ==========================================
# 3. Data Preprocessing 
# ==========================================
initial_rows = len(df)

#  Handle Missing Data (NaNs)
missing_before = df[features + [target]].isnull().sum().sum()
df = df.dropna(subset=features + [target])
if missing_before > 0:
    print(f" 1. Missing Data Handling: Removed rows containing missing (NaN) values.")
else:
    print(" 1. Missing Data Handling: Checked, no missing values detected.")

#  Data Deduplication
rows_before_dedup = len(df)
df = df.drop_duplicates()
duplicates_removed = rows_before_dedup - len(df)
print(f" 2. Data Deduplication: Removed {duplicates_removed} duplicate records to prevent model bias.")

#  Outlier Removal using IQR (Interquartile Range)
# We remove extreme historical yield values that could confuse the model
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

rows_before_outliers = len(df)
df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]
outliers_removed = rows_before_outliers - len(df)
print(f" 3. Outlier Removal (IQR Method): Removed {outliers_removed} extreme outliers.")

print(f" -> Final clean dataset ready for training: {len(df)} records.")

# Separate inputs (X) and answers (y)
X = df[features]
y = df[target]

# Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. Model Training 
# ==========================================
print(f"\n---  Training Decision Tree Regressor for {target_crop.upper()} ---")
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 5. Model Evaluation
# ==========================================
y_pred = model.predict(X_test)

mae = np.mean(np.abs(y_test - y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f" - R-Squared Score: {r2:.4f} (Higher is better, max is 1.0)")
print(f" - Mean Absolute Error: {mae:.2f} kg/ha")

# ==========================================
# 6. Recommendation Logic (The 'Optimizer')
# ==========================================
def get_fertilizer_recommendation(predicted_yield):
    if predicted_yield < 1000: 
        return "Recommendation: INCREASE fertilizer application. Soil nutrients are likely insufficient."
    elif predicted_yield > 2500: 
        return "Recommendation: MAINTAIN current levels. Yield is optimal for this region."
    else:
        return "Recommendation: MONITOR soil health. Consider slight adjustments."

# ==========================================
# 7. Test Prediction (Proof of Concept)
# ==========================================
print("\n---  PoC Demonstration (Sample Input) ---")

sample_input = [[20, 20, 20,6.5, 600]] 
predicted_val = model.predict(sample_input)[0]

print(f"Target District      : {target_district.upper()}")
print(f"Target Crop          : {target_crop.upper()}")
print("-" *60)
print(f"Input Soil Data      : {sample_input}")
print(f"Predicted Crop Yield : {predicted_val:.2f} kg/ha")
print(get_fertilizer_recommendation(predicted_val))

print("\nProof of Concept Execution Complete.")  