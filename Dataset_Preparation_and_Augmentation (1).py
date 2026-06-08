import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================================================================
# 1. Load data
# =========================================================================
X_data = np.load('X_data.npy')
y_data = np.load('y_data.npy')

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_data)

# =========================================================================
# 2. Split the data
# =========================================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# =========================================================================
# 3. Advanced Spatial Augmentation Function (Without Slicing)
# =========================================================================
def augment_to_target(X, y, target_count=1200):
    """
    Applies Rotation, Scaling, Flipping, and Noise to MediaPipe landmarks.
    Loops until target_count is met, keeping ALL generated samples from the last loop.
    """
    X_aug_list = [X]
    y_aug_list = [y]
    
    current_count = X.shape[0]
    print(f"Starting advanced augmentation. Current: {current_count}, Target Minimum: {target_count}")
    
    N, T, F = X.shape
    num_landmarks = F // 3  # 126 // 3 = 42 landmarks (x, y, z)
    
    while current_count < target_count:
        # Reshape to (N, 30, 42, 3) to easily manipulate X, Y, Z coordinates
        X_spatial = X.reshape(N, T, num_landmarks, 3)
        
        # --- 1. ROTATION (Around Z-axis / Image Plane) ---
        theta = np.radians(np.random.uniform(-10, 10))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])
        X_rotated = np.einsum('ntji,ki->ntjk', X_spatial, R).reshape(N, T, F)
        
        # --- 2. SCALING & NOISE ---
        scale_factor = np.random.uniform(0.9, 1.1)
        noise = np.random.normal(0, np.random.uniform(0.003, 0.01), X.shape)
        X_scaled = (X * scale_factor) + noise
        
        # --- 3. HORIZONTAL FLIPPING (Invert X-axis for relative coordinates) ---
        X_flipped_spatial = X_spatial.copy()
        X_flipped_spatial[:, :, :, 0] = -X_flipped_spatial[:, :, :, 0]
        X_flipped = X_flipped_spatial.reshape(N, T, F)
        
        # Append all new variations
        X_aug_list.extend([X_rotated, X_scaled, X_flipped])
        y_aug_list.extend([y, y, y])
        
        # Update current count
        current_count += (N * 3)
    
    X_final = np.concatenate(X_aug_list, axis=0)
    y_final = np.concatenate(y_aug_list, axis=0)
    
    return X_final, y_final

# =========================================================================
# 4. Generate the dataset
# =========================================================================
print("--- Augmenting Training Set ---")
X_train_augmented, y_train_augmented = augment_to_target(X_train, y_train, target_count=1200)

print("\n--- Augmenting Validation Set ---")
X_val_augmented, y_val_augmented = augment_to_target(X_val, y_val, target_count=100)

print("\n--- Augmenting Testing Set ---")
X_test_augmented, y_test_augmented = augment_to_target(X_test, y_test, target_count=100)

# =========================================================================
# 5. Save the prepared data locally
# =========================================================================
save_dir = "model_ready_data" 
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "X_train.npy"), X_train_augmented)
np.save(os.path.join(save_dir, "y_train.npy"), y_train_augmented)

np.save(os.path.join(save_dir, "X_val.npy"), X_val_augmented)
np.save(os.path.join(save_dir, "y_val.npy"), y_val_augmented)

np.save(os.path.join(save_dir, "X_test.npy"), X_test_augmented)
np.save(os.path.join(save_dir, "y_test.npy"), y_test_augmented)

print("\n--- Augmentation Complete ---")
print(f"Successfully saved datasets in: {save_dir}")
print(f"Final X_train shape: {X_train_augmented.shape}")
print(f"Final X_val shape: {X_val_augmented.shape}")
print(f"Final X_test shape: {X_test_augmented.shape}")