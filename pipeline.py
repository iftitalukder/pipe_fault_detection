import os
import time
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout

# -----------------------------
# PARAMETERS
# -----------------------------
window_size = 50
step_size = 25
strong_noise_std = 0.4
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv('balanced_dataset.csv')  # replace with your CSV
X_raw = data[['X','Y','Z']].values
y_raw = data['class'].values

# -----------------------------
# CREATE WINDOWS
# -----------------------------
X_windows, y_windows = [], []
for start in range(0, len(X_raw) - window_size + 1, step_size):
    X_windows.append(X_raw[start:start+window_size])
    y_windows.append(np.bincount(y_raw[start:start+window_size]).argmax())  # majority vote
X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

# -----------------------------
# TRAIN-VAL-TEST SPLIT
# -----------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_windows, y_windows, test_size=0.2, random_state=42, stratify=y_windows
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval
)

# -----------------------------
# SCALE DATA
# -----------------------------
n_samples, n_timesteps, n_axes = X_train.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_axes)).reshape(n_samples, n_timesteps, n_axes)
X_val_scaled = scaler.transform(X_val.reshape(-1, n_axes)).reshape(X_val.shape[0], n_timesteps, n_axes)
X_test_scaled = scaler.transform(X_test.reshape(-1, n_axes)).reshape(X_test.shape[0], n_timesteps, n_axes)

# -----------------------------
# TRAIN CLASSICAL MODELS
# -----------------------------
X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)

classical_models = [
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC(kernel='rbf', probability=True, random_state=42)),
    ('LogisticRegression', LogisticRegression(max_iter=500, random_state=42)),
    ('kNN', KNeighborsClassifier(n_neighbors=5))
]

results = []
for name, model in classical_models:
    start_time = time.time()
    model.fit(X_train_flat, y_train)
    train_time = time.time() - start_time

    # Clean test
    y_pred_clean = model.predict(X_test_flat)
    acc_clean = accuracy_score(y_test, y_pred_clean)*100
    report_clean = classification_report(y_test, y_pred_clean, output_dict=True)
    cm_clean = confusion_matrix(y_test, y_pred_clean)

    # Noisy test
    X_test_noisy_flat = X_test_flat + np.random.normal(0, strong_noise_std, X_test_flat.shape)
    y_pred_noisy = model.predict(X_test_noisy_flat)
    acc_noisy = accuracy_score(y_test, y_pred_noisy)*100
    report_noisy = classification_report(y_test, y_pred_noisy, output_dict=True)
    cm_noisy = confusion_matrix(y_test, y_pred_noisy)

    results.append({
        'Model': name,
        'Clean Acc (%)': acc_clean,
        'Noisy Acc (%)': acc_noisy,
        'Clean Precision': np.mean([v['precision'] for k,v in report_clean.items() if k.isdigit()]),
        'Noisy Precision': np.mean([v['precision'] for k,v in report_noisy.items() if k.isdigit()]),
        'Clean Recall': np.mean([v['recall'] for k,v in report_clean.items() if k.isdigit()]),
        'Noisy Recall': np.mean([v['recall'] for k,v in report_noisy.items() if k.isdigit()]),
        'Clean F1': np.mean([v['f1-score'] for k,v in report_clean.items() if k.isdigit()]),
        'Noisy F1': np.mean([v['f1-score'] for k,v in report_noisy.items() if k.isdigit()]),
        'Clean Confusion': str(cm_clean.tolist()),
        'Noisy Confusion': str(cm_noisy.tolist()),
        'Train+Infer Time (s)': train_time
    })

# -----------------------------
# TRAIN CNN-LSTM
# -----------------------------
cnn_lstm = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_axes)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y_raw)), activation='softmax')
])
cnn_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
cnn_lstm.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_val_scaled, y_val), verbose=2)
train_time_cnn = time.time() - start_time

# Clean test
y_pred_clean_cnn = np.argmax(cnn_lstm.predict(X_test_scaled), axis=1)
acc_clean_cnn = accuracy_score(y_test, y_pred_clean_cnn)*100
report_clean_cnn = classification_report(y_test, y_pred_clean_cnn, output_dict=True)
cm_clean_cnn = confusion_matrix(y_test, y_pred_clean_cnn)

# Noisy test
X_test_noisy = X_test_scaled + np.random.normal(0, strong_noise_std, X_test_scaled.shape)
y_pred_noisy_cnn = np.argmax(cnn_lstm.predict(X_test_noisy), axis=1)
acc_noisy_cnn = accuracy_score(y_test, y_pred_noisy_cnn)*100
report_noisy_cnn = classification_report(y_test, y_pred_noisy_cnn, output_dict=True)
cm_noisy_cnn = confusion_matrix(y_test, y_pred_noisy_cnn)

results.append({
    'Model': 'CNN-LSTM',
    'Clean Acc (%)': acc_clean_cnn,
    'Noisy Acc (%)': acc_noisy_cnn,
    'Clean Precision': np.mean([v['precision'] for k,v in report_clean_cnn.items() if k.isdigit()]),
    'Noisy Precision': np.mean([v['precision'] for k,v in report_noisy_cnn.items() if k.isdigit()]),
    'Clean Recall': np.mean([v['recall'] for k,v in report_clean_cnn.items() if k.isdigit()]),
    'Noisy Recall': np.mean([v['recall'] for k,v in report_noisy_cnn.items() if k.isdigit()]),
    'Clean F1': np.mean([v['f1-score'] for k,v in report_clean_cnn.items() if k.isdigit()]),
    'Noisy F1': np.mean([v['f1-score'] for k,v in report_noisy_cnn.items() if k.isdigit()]),
    'Clean Confusion': str(cm_clean_cnn.tolist()),
    'Noisy Confusion': str(cm_noisy_cnn.tolist()),
    'Train+Infer Time (s)': train_time_cnn
})

# -----------------------------
# SAVE SUMMARY CSV
# -----------------------------
summary_df = pd.DataFrame(results)
summary_csv = os.path.join(save_dir, "vibration_model_results_readable.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"✅ Summary CSV saved at: {summary_csv}")

# -----------------------------
# SAVE CONFUSION MATRICES AS CSVS
# -----------------------------
for i, row in summary_df.iterrows():
    model_name = row['Model']
    cm_clean = np.array(literal_eval(row['Clean Confusion']))
    cm_noisy = np.array(literal_eval(row['Noisy Confusion']))
    pd.DataFrame(cm_clean).to_csv(os.path.join(save_dir, f"{model_name}_Clean_ConfMatrix.csv"), index=True, header=True)
    pd.DataFrame(cm_noisy).to_csv(os.path.join(save_dir, f"{model_name}_Noisy_ConfMatrix.csv"), index=True, header=True)

print(f"✅ All confusion matrices saved as CSVs in '{save_dir}'")
