import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
test_dir = r'C:\Users\sanja\OneDrive\Desktop\FARM AI\test'
train_dir = r'C:\Users\sanja\OneDrive\Desktop\FARM AI\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'
model_path = 'best_model.keras'

# Load model
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Model file '{model_path}' not found.")
    exit()

# Data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='sparse',
    shuffle=False
)

# Label info
train_labels = sorted(os.listdir(train_dir))
test_labels = sorted(test_generator.class_indices.keys())

print(f"📚 Train Labels: {train_labels}")
print(f"🧪 Test Labels: {test_labels}")

# Mapping check
common_labels = list(set(train_labels) & set(test_labels))
print(f"✅ Common Labels: {common_labels}")

# Predictions
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Check for mismatch
if len(predicted_classes) != len(true_classes):
    print(f"❌ Mismatch: {len(predicted_classes)} predictions vs {len(true_classes)} true classes")
    exit()

# Print results
print("\n📸 Prediction Results:")
for i, filepath in enumerate(test_generator.filepaths):
    pred_idx = predicted_classes[i]
    true_idx = true_classes[i]

    pred_label = train_labels[pred_idx] if pred_idx < len(train_labels) else "Out of range"
    true_label = test_labels[true_idx] if true_idx < len(test_labels) else "Unknown"

    print(f"🖼️ {os.path.basename(filepath)}")
    print(f"   ✅ True Label:      {true_label}")
    print(f"   🔍 Predicted Label: {pred_label}")
    print("-" * 40)

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=train_labels, yticklabels=train_labels)
plt.title("🔍 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=train_labels))
