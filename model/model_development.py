# model/model_development.py

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import joblib
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the Wine Dataset
print("Loading Wine Dataset...")
wine_data = load_wine()

# Step 2: Convert to pandas DataFrame
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df['cultivar'] = wine_data.target

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

# Step 3: Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")
print("No missing values found!")

# Step 4: Feature Selection - Select 6 features
selected_features = [
    'alcohol',
    'flavanoids',
    'color_intensity',
    'hue',
    'od280/od315_of_diluted_wines',
    'proline'
]

print(f"\nSelected Features: {selected_features}")

# Extract features and target
X = df[selected_features]
y = df['cultivar']

# Step 5: Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Step 6: Feature Scaling (Mandatory)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")

# Step 7: Train the Model - Random Forest Classifier
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5
)

model.fit(X_train_scaled, y_train)
print("Model training completed!")

# Step 8: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 9: Evaluate the Model
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Precision, Recall, F1-Score
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average='weighted'
)

print(f"\nWeighted Metrics:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Full Classification Report
print(f"\nDetailed Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Cultivar 1', 'Cultivar 2', 'Cultivar 3']
))

# Step 10: Save the Model and Scaler
print("\nSaving model and scaler...")
joblib.dump(model, 'wine_cultivar_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

print("\n✅ Model saved as 'wine_cultivar_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")
print("✅ Features saved as 'selected_features.pkl'")
print("\nModel development completed successfully!")
