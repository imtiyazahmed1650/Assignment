# Step 1: Import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset
df = pd.read_excel("DemoWaterLevel.xlsx")   # update filename if needed

# Step 3: Define dependent and independent variables
y = df['dataValue']

features = [
    'Year', 'Month_cos', 'Month_sin', 'Day_cos', 'Day_sin',
    'latitude', 'longitude', 'wellDepth',
    'Value_atm_pre', 'Value_Evp', 'Value_Rain', 'Value_Temp',
    'Value_Sm1', 'Value_Sm2', 'Value_Sm3', 'Value_Sm4'
]
#this is to test changeS
# Changes 2

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

# Step 5: Add constant for intercept
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Step 6: Fit MLR model using statsmodels
model = sm.OLS(y_train, X_train_sm).fit()

# Step 7: Print model summary (coefficients, p-values, R², AIC/BIC)
print(model.summary())

# Step 8: Predictions
y_pred = model.predict(X_test_sm)

# Step 9: Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print("RMSE:", rmse)
print("R² Score:", r2)

# Step 10: Residual Plot (check assumptions)
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Step 11: Correlation Heatmap (EDA check)
plt.figure(figsize=(12,8))
sns.heatmap(df[features + ['dataValue']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap with Groundwater Level")
plt.show()
