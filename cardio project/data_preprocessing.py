import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Define dataset path
file_path = r'C:\Users\babum\OneDrive\Desktop\cardio project\heart_disease_data.csv'

# ✅ Check if the file exists before proceeding
if not os.path.exists(file_path):
    print(f"❌ ERROR: File not found at {file_path}. Please check the path!")
    exit()

# ✅ Load the dataset
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!")

# ✅ Handle missing values (if any)
df.dropna(inplace=True)  # Removes rows with missing values

# ✅ Split features (X) and target (y)
if 'target' not in df.columns:
    print("❌ ERROR: 'target' column not found in dataset. Please check column names.")
    exit()

X = df.drop(columns=['target'])  # Features
y = df['target']  # Labels

# ✅ Split into train-test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Normalize features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Save processed data for training
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("✅ Data preprocessing complete. Processed files saved!")
