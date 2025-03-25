import pandas as pd
import numpy as np
import warnings
from math import sqrt
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from rulefit import RuleFit
import joblib

warnings.filterwarnings("ignore")

# 📌 **1. Load Training Data**
print("📌 Loading dataset...")
train_data = pd.read_csv("extra_data_ready.csv")

# 📌 **2. Define Target Variable & Features**
label = "Sigma"
categorical_features = ["Type"]  # Update with actual categorical columns

# 📌 **3. One-Hot Encode Categorical Features**
encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_train = pd.DataFrame(encoder.fit_transform(train_data[categorical_features]))
encoded_train.columns = encoder.get_feature_names_out(categorical_features)

# 📌 **4. Feature Selection**
autogluon_148=['nBase', 'AATS3m', 'PEOE_VSA1', 'nH', 'IC0', 'AATS4pe', 'AATS4se', 'BCUTd-1l', 'AATS3Z', 'BCUTi-1h', 'SdO', 'AATS3p', 'JGI2', 'GATS1Z', 'nAcid', 'ETA_beta_ns', 'BCUTs-1l', 'PEOE_VSA7', 'Xc-5dv', 'Xc-5d', 'PEOE_VSA10', 'AATS4s', 'AATS3v', 'BCUTm-1l', 'GATS6s', 'MINaaCH', 'BCUTZ-1l', 'GATS6se', 'AATS1m', 'GATS6are', 'PEOE_VSA9', 'AATSC0d', 'AATS5pe', 'AATSC1d', 'GATS2v', 'AMID_h', 'MID_h', 'AMID_N', 'AATS4Z', 'GATS4se', 'AATS1Z', 'GATS6pe', 'AETA_eta_L', 'NtN', 'AATS4m', 'SlogP_VSA4', 'SMR_VSA2', 'MIC3', 'VSA_EState5', 'MATS1pe', 'MATS1se', 'BCUTi-1l', 'GATS1dv', 'GATS2s', 'ATSC2v', 'MIC2', 'GATS1m', 'SIC0', 'MID_N', 'BCUTZ-1h', 'MINdssC', 'VSA_EState4', 'AETA_dBeta', 'SM1_Dzse', 'nBondsD', 'ETA_epsilon_1', 'AATSC2v', 'AATS4v', 'BIC5', 'TPSA', 'BCUTare-1l', 'Xch-7d', 'AETA_eta', 'AATS2v', 'Mv', 'nBondsM', 'BCUTpe-1l', 'AATS1v', 'AATS3dv', 'NssCH2', 'MAXaasC', 'BIC4', 'GATS4dv', 'GATS1are', 'ETA_dEpsilon_D', 'MATS1p', 'MAXdO', 'AATSC1Z', 'ATSC1pe', 'TopoPSA', 'GATS4s', 'LogP', 'nP', 'ATSC1Z', 'SaasC', 'GATS1pe', 'PEOE_VSA6', 'SMR_VSA1', 'AATSC1pe', 'BCUTm-1h', 'AATS1d', 'ATSC1se', 'VSA_EState2', 'BCUTse-1l', 'PEOE_VSA11', 'StN', 'GATS4pe', 'PEOE_VSA8', 'ATSC3d', 'AATSC1se', 'NddssS', 'MATS2s', 'AATS5v', 'GATS2se', 'ATSC7p', 'GATS5dv', 'AATS3s', 'MATS6se', 'VSA_EState3', 'Xpc-4d', 'AATSC0p', 'nBondsKD', 'SlogP_VSA1', 'AETA_beta', 'MZ', 'SMR', 'IC2', 'GATS1i', 'MDEO-11', 'AATSC1dv', 'MATS6are', 'AATS7s', 'ATSC5s', 'AATSC1m', 'SdssC', 'ATSC0p', 'GATS1se', 'GATS4are', 'AATS2m', 'AATS2i', 'GATS3p', 'Xch-7dv', 'piPC6', 'GATS2are', 'ATSC7i', 'ATSC6s', 'AETA_beta_ns', 'SlogP_VSA10', ]
k_fold_feature_selection = autogluon_148

# 📌 **5. Combine Categorical & Numerical Features**
train_data = train_data.drop(columns=categorical_features)
train_data = pd.concat([train_data, encoded_train], axis=1)

# 📌 **6. Select Features**
train_df = train_data.reindex(columns=k_fold_feature_selection, fill_value=0)

# 📌 **7. Split Features & Target**
X = train_df
y = train_data[label]

# 📌 **8. Feature Scaling**
scaler_X = MinMaxScaler()
X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)

# 📌 **9. Target Scaling**
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1)).flatten()

# 📌 **10. Handle Missing Values**
imputer = SimpleImputer(strategy="median")
X_scaled = imputer.fit_transform(X_scaled)

# 📌 **11. Split into Train & Validation Sets**
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=42)

# 📌 **12. Train RuleFit Model**
print("📌 Training RuleFit model...")
rulefit_model = RuleFit(tree_size=20, rfmode="regress", lin_standardise=True, max_iter=20000, random_state=42)
rulefit_model.fit(X_train, y_train, feature_names=k_fold_feature_selection)
print("✅ Training finished.")

# 📌 **13. Save Model**
joblib.dump(rulefit_model, "rulefit_model.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
joblib.dump(imputer, "imputer.pkl")
print("📌 Model and scalers saved successfully.")


# 📌 **13. Cross-Validation R²**
cross_val_r2 = np.mean(cross_val_score(rulefit_model, X_train, y_train, cv=5, scoring="r2"))

# 📌 **14. Predict on Train & Validation Set**
y_train_pred_scaled = rulefit_model.predict(X_train)
y_test_pred_scaled = rulefit_model.predict(X_val)

# Convert predictions back to original scale
y_train_pred_original = scaler_y.inverse_transform(np.array(y_train_pred_scaled).reshape(-1, 1)).flatten()
y_test_pred_original = scaler_y.inverse_transform(np.array(y_test_pred_scaled).reshape(-1, 1)).flatten()
y_train_original = scaler_y.inverse_transform(np.array(y_train).reshape(-1, 1)).flatten()
y_val_original = scaler_y.inverse_transform(np.array(y_val).reshape(-1, 1)).flatten()

# 📌 **15. Evaluate Model Performance**
r2_train = r2_score(y_train_original, y_train_pred_original)
r2_test = r2_score(y_val_original, y_test_pred_original)
rmse_test = sqrt(mean_squared_error(y_val_original, y_test_pred_original))

# 📌 **16. Print Model Performance**
print("\n📌 Model Performance Summary:")
print(f"➡ Cross-Val R²: {cross_val_r2:.4f}")
print(f"➡ Train R²: {r2_train:.4f}")
print(f"➡ Test R²: {r2_test:.4f}")
print(f"➡ Test RMSE: {rmse_test:.4f}")
