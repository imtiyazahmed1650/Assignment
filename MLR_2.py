# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Import Random Forest Regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Step 2: Load dataset
df = pd.read_csv("groundwater_selected_columns.csv")

# Preprocessing: Convert 'date' and create time-based features
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
df['Month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['Day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)
df['Day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)

# Step 3: Define dependent and independent variables
y = df['dataValue']

# Using the refined feature list
features = [
    'Year', 'Month_cos', 'Month_sin', 'Day_cos', 'Day_sin',
    'latitude', 'longitude', 'wellDepth',
    'total_population_count',
    'area_irrig',
    'Stage of Ground Water Extraction (%)_2023',
    'sand_0-5cm_mean',
    'silt_0-5cm_mean',
]

X = df[features]

# Handle NaN and Inf
X = X.replace([np.inf, -np.inf], np.nan)
df_clean = pd.concat([X, y], axis=1).dropna()

# Updated clean X and y
X = df_clean[features]
y = df_clean['dataValue']

# Step 4: Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Fit Random Forest Regressor model
# A Random Forest model is an excellent drop-in replacement
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Step 6: Create output folder (if it doesn't exist)
output_dir = "groundwater_analysis_results_rf"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Save evaluation metrics to a text file
with open(os.path.join(output_dir, 'evaluation_metrics_rf.txt'), 'w') as f:
    f.write("Evaluation Metrics (Random Forest):\n")
    f.write(f"R² Score: {r2}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"MAE: {mae}\n")

print("\nEvaluation Metrics saved to file.")

# Step 9: Plot and save Feature Importance
plt.figure(figsize=(10, 8))
importance = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importance)[::-1]
sns.barplot(x=importance[sorted_idx], y=feature_names[sorted_idx])
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
print("Feature Importance plot saved to file.")
plt.show()

# Step 10: Residual Plot (check assumptions)
plt.figure(figsize=(6, 4))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Random Forest)")
plt.savefig(os.path.join(output_dir, 'residual_plot_rf.png'))
print("Residual Plot saved to file.")

# Step 11: Correlation Heatmap (EDA check)
plt.figure(figsize=(12, 8))
sns.heatmap(df_clean[features + ['dataValue']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap with Groundwater Level")
plt.savefig(os.path.join(output_dir, 'correlation_heatmap_rf.png'))
print("Correlation Heatmap saved to file.")
plt.show()# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Import Random Forest Regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Step 2: Load dataset
df = pd.read_csv("groundwater_selected_columns.csv")

# Preprocessing: Convert 'date' and create time-based features
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
df['Month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['Day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)
df['Day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)

# Step 3: Define dependent and independent variables
y = df['dataValue']

# Using the refined feature list
features = [
    'Year', 'Month_cos', 'Month_sin', 'Day_cos', 'Day_sin',
    'latitude', 'longitude', 'wellDepth',
    'total_population_count',
    'area_irrig',
    'Stage of Ground Water Extraction (%)_2023',
    'sand_0-5cm_mean',
    'silt_0-5cm_mean',
]

X = df[features]

# Handle NaN and Inf
X = X.replace([np.inf, -np.inf], np.nan)
df_clean = pd.concat([X, y], axis=1).dropna()

# Updated clean X and y
X = df_clean[features]
y = df_clean['dataValue']

# Step 4: Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Fit Random Forest Regressor model
# A Random Forest model is an excellent drop-in replacement
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Step 6: Create output folder (if it doesn't exist)
output_dir = "groundwater_analysis_results_rf"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Save evaluation metrics to a text file
with open(os.path.join(output_dir, 'evaluation_metrics_rf.txt'), 'w') as f:
    f.write("Evaluation Metrics (Random Forest):\n")
    f.write(f"R² Score: {r2}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"MAE: {mae}\n")

print("\nEvaluation Metrics saved to file.")

# Step 9: Plot and save Feature Importance
plt.figure(figsize=(10, 8))
importance = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importance)[::-1]
sns.barplot(x=importance[sorted_idx], y=feature_names[sorted_idx])
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
print("Feature Importance plot saved to file.")
plt.show()

# Step 10: Residual Plot (check assumptions)
plt.figure(figsize=(6, 4))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Random Forest)")
plt.savefig(os.path.join(output_dir, 'residual_plot_rf.png'))
print("Residual Plot saved to file.")

# Step 11: Correlation Heatmap (EDA check)
plt.figure(figsize=(12, 8))
sns.heatmap(df_clean[features + ['dataValue']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap with Groundwater Level")
plt.savefig(os.path.join(output_dir, 'correlation_heatmap_rf.png'))
print("Correlation Heatmap saved to file.")
plt.show()