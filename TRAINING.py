import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('mixed_float16')

# Paths
train_dir = r'C:\Users\sanja\OneDrive\Desktop\FARM AI\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'
val_dir = r'C:\Users\sanja\OneDrive\Desktop\FARM AI\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid'
checkpoint_path = "epoch_checkpoints/model_epoch_{epoch:02d}.weights.h5"

# Create checkpoint directory
os.makedirs("epoch_checkpoints", exist_ok=True)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

# Base model
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True
for layer in base_model.layers[:50]:
    layer.trainable = False

# Full model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(train_generator.class_indices), activation='softmax', dtype='float32')
])

model.compile(
    optimizer=optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Resume logic
latest = tf.train.latest_checkpoint("epoch_checkpoints")
initial_epoch = 0
if latest:
    print(f"🔁 Resuming from checkpoint: {latest}")
    model.load_weights(latest)
    initial_epoch = int(latest.split('_')[-1].split('.')[0])
else:
    print("🆕 No previous checkpoint found. Starting fresh.")

# Time tracking
total_start_time = time.time()

# Epoch-wise time logger
class TimeLogger(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = (time.time() - self.epoch_start) / 60
        print(f"⏱️ Epoch {epoch + 1} took {duration:.2f} minutes.")

# Callbacks
callback_list = [
    callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True),
    callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6),
    TimeLogger()
]

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    validation_data=val_generator,
    validation_steps=40,
    epochs=20,
    initial_epoch=initial_epoch,
    callbacks=callback_list
)

# Total time
total_elapsed_time = (time.time() - total_start_time) / 3600
print(f"\n✅ Total training time: {total_elapsed_time:.2f} hours.")

# Evaluation
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

print("📊 Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=val_generator.class_indices.keys()))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_true, y_pred_classes))

# Accuracy & Loss Plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()
