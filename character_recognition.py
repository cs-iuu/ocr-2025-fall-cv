import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

DATA_DIR = "data/characters"
IMG_SIZE = (64, 64)
BATCH_SIZE = 8
EPOCHS = 10

for root, dirs, files in os.walk(DATA_DIR):
    for d in dirs:
        if d.startswith("."):
            os.rmdir(os.path.join(root, d))
    for f in files:
        if f.startswith("."):
            os.remove(os.path.join(root, f))
            
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)


model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(train_ds, epochs=EPOCHS)

y_true = []
y_pred = []

for images, labels in train_ds:
    predictions = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

macro_f1 = f1_score(y_true, y_pred, average='macro')

print("\n Evaluation Results")
print("----------------------")
print("Macro F1-score:", round(macro_f1, 4))
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))


model.save("char_cnn_model.keras")
print("\n✅ Model saved as char_cnn_model.keras")
