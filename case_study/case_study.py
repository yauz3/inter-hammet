#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/02/2025
# Author: Sadettin Y. Ugurlu


import pandas as pd
import numpy as np
import warnings
from math import sqrt
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from rulefit import RuleFit

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


# 📌 **13. Cross-Validation R²**
cross_val_r2 = np.mean(cross_val_score(rulefit_model, X_train, y_train, cv=5, scoring="r2"))

# 📌 **14. Predict on Train & Validation Set**
y_train_pred_scaled = rulefit_model.predict(X_train)
y_test_pred_scaled = rulefit_model.predict(X_val)

# 📌 **14.5: Convert predictions back to original scale**
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

# 📌 **17. Print Predictions vs Actuals**
print("🎯 True values:")
print(y_val_original)

print("🔮 Predicted values:")
print(y_test_pred_original)


"""
"🎯 True values:"
y_true=[
    0.09, 0.38, 0.04, 0.45, 0.33, 0.44, 0.58, 0.49, 0.21, 0.85, 0.49, 0.57,
    0.66, 0.10, 1.09, -0.04, 0.07, -0.03, 0.64, -0.07, 0.30, 0.45, 1.11, 0.59,
    -0.13, 0.92, 0.39, 0.26, 0.24, 0.38, 0.38, 0.62, 0.29, 0.51, 0.50, 0.00,
    0.05, 0.26, 0.76, 0.34, 0.17, -0.04, 0.72, 0.39, 0.54, 0.31, 0.00, 0.25,
    0.12, 0.29, -1.21, 0.17, -0.07, 0.40, 0.40, 0.01, 0.31, 0.43, 0.00, -0.14,
    0.46, 0.33, 1.24, -0.27, 0.63, 0.92, 0.18, 0.12, 1.76, 0.08, 0.33, 0.38,
    0.28, 0.15, 0.31, 0.06, 0.12, 0.53, 0.21, 0.02, 0.24, 0.16, 0.00, 0.10,
    0.66, 0.15, -0.17, 0.10, 1.10, 0.21, 0.28, 0.02, 0.23, 0.12, 0.43, 0.09,
    0.65, 0.12, -0.47, 0.78, 0.21, -0.07, -0.81, 0.47, 0.45, 0.19, 0.46, -0.03,
    0.67, 0.39, 0.52, 0.15, 0.10, -0.07, -0.09, -0.02, 0.60, 0.23, 0.31, -0.22,
    0.30, 0.02, -0.08, 0.40, 1.13, -0.07, 0.61, 0.44, -0.17, -0.01, 0.17, 0.80,
    0.56, 0.72, 0.00, 0.24, 0.96, 0.96, 0.23, -0.07, -0.06, -0.01, 0.09, 0.23,
    0.83, 0.30, -0.16, 0.12
]


🔮 Predicted values:
y_predicted=[
    0.0998180748, 0.241239697, 0.0457403979, 0.441562937, 0.325993656,
    0.448492417, 0.519741863, 0.460605427, 0.119366203, 0.852217557,
    0.371994814, 0.613223046, 0.695832809, 0.107176618, 1.11034512,
    0.0210562692, 0.247942501, 0.0707046572, 0.492529566, -0.102621999,
    0.192288217, 0.392094541, 0.975234475, 0.586731998, -0.0972482913,
    1.05157936, 0.505054277, 0.18740601, 0.278101508, 0.46396373,
    0.357202244, 0.536014658, 0.248226041, 0.663170422, 0.572983797,
    0.0722592274, -0.075409578, -0.035365403, 0.803534214, 0.183776238,
    0.231425771, -0.0213779742, 0.717255239, 0.378268997, 0.598401113,
    0.273243099, 0.0258982105, 0.204140052, 0.223039689, 0.141099024,
    -0.42785241, 0.178545493, -0.0857091243, 0.409252607, 0.390642673,
    0.0905393721, 0.318068462, 0.290646729, 0.0881277192, -0.158626112,
    0.530532771, 0.360544592, 1.33253491, -0.202034517, 0.443249408,
    1.03572315, 0.359064761, 0.137332956, 1.40593997, 0.0352076488,
    0.340333628, 0.425432005, 0.355011198, 0.199410601, 0.281111295,
    -0.023559794, -0.0232829842, 0.64641314, 0.160311796, 0.067209554,
    0.110098753, 0.0931795997, -0.22953869, 0.0549943378, 0.64927415,
    0.207993093, -0.191940187, 0.120387173, 0.882653384, 0.251919744,
    0.403279567, 0.0131802592, 0.198619249, 0.205911488, 0.116308119,
    0.153229711, 0.342290876, 0.0711453685, -0.251952953, 0.741117915,
    0.268960442, 0.000561340241, -0.433130966, 0.365989493, 0.394833209,
    0.164736831, 0.487916613, -0.190808013, 0.524834036, 0.396046056,
    0.669379098, 0.102746008, 0.0582315812, -0.0068893857, -0.0629900841,
    -0.0209912484, 0.165347616, 0.190410217, 0.540939475, 0.0023708788,
    0.225941973, -0.0121877485, 0.000832426475, 0.429710969, 1.06426923,
    -0.0961349008, 0.554241342, 0.245307478, -0.145920111, -0.0887258456,
    0.137867409, 0.784586052, 0.565310045, 0.745664618, 0.130479089,
    0.181977248, 0.865676808, 0.920818326, 0.275173853, 0.062493049,
    -0.0295992444, 0.0728154696, 0.0659828293, 0.267352341, 0.727315063,
    0.200251749, -0.145613012, 0.06684058
]
"""
