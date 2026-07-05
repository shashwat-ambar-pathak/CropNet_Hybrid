import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional, reduces some warnings
import numpy as np
import tensorflow as tf
import pickle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import load_model
from cropnet_hybrid_model import build_cropnet_hybrid

# ----------------- CPU Optimization -----------------
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

# ----------------- Paths & Parameters -----------------
train_dir = "train"
val_dir = "val"
test_dir = "test"
img_size = (224, 224)
batch_size = 16  # smaller batch size for CPU
num_classes = 38
epochs = 36             # <-- total epochs now set to 40
resume_model_path = "cropnet_hybrid_model_final.keras"
previous_epochs = 32       # <-- you already completed 18/25

# ----------------- Data Generators -----------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_gen = ImageDataGenerator(rescale=1./255)
val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ----------------- Class Weights -----------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("✅ Class Weights:", class_weight_dict)

# ----------------- Load or Build Model -----------------
if os.path.exists(resume_model_path):
    print(f"⏳ Loading existing model from {resume_model_path} to resume training...")
    model = load_model(resume_model_path, compile=True,
                       custom_objects={"top3_acc": TopKCategoricalAccuracy(k=3)})
    initial_epoch = previous_epochs
    print(f"➡️ Resuming from epoch {initial_epoch+1}/{epochs}")
else:
    print("⏳ Building new CropNet hybrid model...")
    model = build_cropnet_hybrid(input_shape=(224, 224, 3), num_classes=num_classes)
    initial_epoch = 0

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3, name="top3_acc")]
)

# ----------------- Callbacks -----------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    ModelCheckpoint(resume_model_path, save_best_only=True, monitor="val_loss", mode="min")
]

# ----------------- Train -----------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    initial_epoch=initial_epoch,   # <-- continue from epoch 19
    callbacks=callbacks,
    class_weight=class_weight_dict,
    shuffle=True
)

# ----------------- Save History -----------------
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("✅ Training history saved")

# ----------------- Evaluate -----------------
test_loss, test_acc, test_top3 = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test Top-3 Accuracy: {test_top3*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ----------------- Save Final Model & Class Indices -----------------
model.save("cropnet_hybrid_model_final.keras")
np.save("class_indices.npy", train_data.class_indices)
print("✅ Final model and class indices saved")

