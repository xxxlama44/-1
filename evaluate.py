import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Define data paths and asset names
DATA_DIR = 'model_ready_data'
MODEL_PATH = 'Emergency_Model_Final.h5'
SCALER_PATH = 'scaler.pkl'

print("Loading validation datasets and model artifacts...")
try:
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    print(f"Error: Missing required files. Details: {e}")
    exit(1)

# Reshape 3D sequence data to 2D for scaling, then back to 3D for LSTM input
n_samples, n_steps, n_features = X_val.shape
X_val_reshaped = X_val.reshape(-1, n_features)
X_val_scaled = scaler.transform(X_val_reshaped).reshape(n_samples, n_steps, n_features)

# Target labels mapping
classes = ['bacteria', 'cancer', 'emergency', 'help', 'hospital', 'need', 'pregnancy', 'swelling', 'virus', 'wound']

# Generate predictions on the validation set
print("Running model inference...")
preds = model.predict(X_val_scaled, verbose=0)
y_pred = np.argmax(preds, axis=1)

# Compute confusion matrix metrics
cm = confusion_matrix(y_val, y_pred)

# Plot confusion matrix heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Emergency Sign Language Recognition', fontsize=14)
plt.ylabel('Actual Labels', fontsize=12)
plt.xlabel('Predicted Labels', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

print("Displaying performance charts...")
plt.show()

# Print text-based classification metrics
print("\n" + "-"*60)
print("CLASSIFICATION PERFORMANCE REPORT")
print("-"*60)
print(classification_report(y_val, y_pred, target_names=classes))
print("-"*60)