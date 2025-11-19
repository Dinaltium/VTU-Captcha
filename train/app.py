"""
VTU Captcha Recognition Training - IMPROVED VERSION
Working with TensorFlow 2.19.0 and Keras 3.10.0
"""

# CELL 1: Setup
import sys, os, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("="*70)
print("VTU CAPTCHA TRAINING - KERAS 3 COMPATIBLE")
print("="*70)
print(f"Python: {sys.version.splitlines()[0]}")
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU Available: {len(gpus)} device(s)")
else:
    print("⚠️  No GPU found - training will be slower")
print("="*70)
print()

# CELL 2: Dataset Configuration
# Dataset: https://www.kaggle.com/datasets/bharatnaik111/vtu-university-captchas-with-labels
# Images are 160x75 (WxH) with noisy background text
# 
# For local training (WSL2/Linux):
#   1. Download dataset from Kaggle manually, OR
#   2. Use Kaggle API: kaggle datasets download -d bharatnaik111/vtu-university-captchas-with-labels
#   3. Extract to: ./data/vtu-university-captchas-with-labels/captchas/
#
# Update DATASET_DIR below to point to your dataset location

# Local dataset path (relative to train/ directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "captchas")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Alternative: Use absolute path
# DATASET_DIR = "/home/rafan/captcha/data/vtu-university-captchas-with-labels/captchas"

# Check if dataset exists
if not os.path.exists(DATASET_DIR):
    print(f"❌ Dataset directory not found: {DATASET_DIR}")
    print("\n📥 To download the dataset:")
    print("   1. Install Kaggle API: pip install kaggle")
    print("   2. Setup Kaggle credentials: ~/.kaggle/kaggle.json")
    print("   3. Run: kaggle datasets download -d bharatnaik111/vtu-university-captchas-with-labels")
    print("   4. Extract to:", DATASET_DIR)
    print("\n   OR download manually from:")
    print("   https://www.kaggle.com/datasets/bharatnaik111/vtu-university-captchas-with-labels")
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

# Check if dataset has enough files
if len(os.listdir(DATASET_DIR)) < 1000:
    print(f"⚠️  Warning: Only {len(os.listdir(DATASET_DIR))} files found in dataset directory")
    print("   Expected at least 1000 files. Dataset may be incomplete.")
else:
    print(f"✓ Dataset found: {DATASET_DIR}")
    print(f"✓ Found {len(os.listdir(DATASET_DIR))} files\n")

# Remove corrupted files if they exist
print("🧹 Checking for corrupted files...")
bad_files = ["2GAUXb.png", "ASkEG4.png", "hB5akC.png", "jTxNsr.png", "RpM4NW.png"]
removed_count = 0
for f in bad_files:
    path = os.path.join(DATASET_DIR, f)
    if os.path.exists(path):
        os.remove(path)
        removed_count += 1
        print(f"  ✓ Removed: {f}")
if removed_count == 0:
    print("  ✓ No corrupted files found")
print()

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"✓ Models will be saved to: {MODELS_DIR}\n")

# CELL 3: Configuration
IMG_HEIGHT = 75  # Fixed: Dataset images are 160x75 (WxH), so height=75
IMG_WIDTH = 160
CAPTCHA_LENGTH = 6
BATCH_SIZE = 128
EPOCHS = 15
EARLY_STOP_PATIENCE = 15

CHARACTERS = "23456789ABCDEFGHJKLMNPRSTUVWXYZabcdefghjkmnprstuvwxyz"
NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank
BLANK_INDEX = NUM_CLASSES - 1

print("CONFIGURATION")
print("="*70)
print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"Captcha length: {CAPTCHA_LENGTH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Characters: {len(CHARACTERS)}")
print(f"Num classes (with blank): {NUM_CLASSES}")
print("="*70)
print()

# CELL 4: Character Mappings
char_to_int = {c: i for i, c in enumerate(CHARACTERS)}
int_to_char = {i: c for i, c in enumerate(CHARACTERS)}

with open(os.path.join(MODELS_DIR, 'char_to_int.json'), 'w') as f:
    json.dump(char_to_int, f, indent=2)
with open(os.path.join(MODELS_DIR, 'int_to_char.json'), 'w') as f:
    json.dump(int_to_char, f, indent=2)

print("✓ Character mappings saved\n")

# CELL 5: Load Dataset
def load_filepaths_and_labels(dataset_dir):
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))
    paths, labels = [], []

    for p in files:
        name = Path(p).stem
        # Keep only desired length and allowed chars
        if len(name) == CAPTCHA_LENGTH and all(ch in CHARACTERS for ch in name):
            paths.append(p)
            labels.append(name)

    return paths, labels

print("LOADING DATASET")
print("="*70)
image_paths, labels = load_filepaths_and_labels(DATASET_DIR)
print(f"✓ Found {len(image_paths)} valid images")

# Show samples
print("\nSample images:")
for i in range(min(5, len(image_paths))):
    print(f"  {Path(image_paths[i]).name} → {labels[i]}")
print("="*70)
print()

# CELL 6: TensorFlow Dataset Pipeline
AUTOTUNE = tf.data.AUTOTUNE

def encode_label(s):
    return [char_to_int[c] for c in s]

encoded_labels = [encode_label(s) for s in labels]
max_label_len = max(len(x) for x in encoded_labels)

def pad_label(lbl):
    # Pad with -1 (interpreted as padding/blank)
    return lbl + [-1] * (max_label_len - len(lbl))

padded_labels = np.array([pad_label(l) for l in encoded_labels], dtype=np.int32)
paths_np = np.array(image_paths)

def _read_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    # Resize to (height, width) - note: tf.image.resize expects [height, width]
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='bilinear', antialias=True)
    # Normalize to [0, 1] range (model has Rescaling layer, but for consistency)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def make_dataset(paths_array, labels_array, batch, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths_array, labels_array))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths_array), seed=42)
    ds = ds.map(lambda p, l: (_read_image(p), l), num_parallel_calls=AUTOTUNE)
    # Format for model: {"image": img, "label": lbl}, None (dummy target)
    ds = ds.map(lambda img, lbl: ({"image": img, "label": lbl}, None), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

# Train/Val split
print("CREATING DATASETS")
print("="*70)
split = int(0.8 * len(paths_np))
train_ds = make_dataset(paths_np[:split], padded_labels[:split], BATCH_SIZE, shuffle=True)
val_ds = make_dataset(paths_np[split:], padded_labels[split:], BATCH_SIZE, shuffle=False)
print(f"✓ Training samples: {split}")
print(f"✓ Validation samples: {len(paths_np) - split}")
print("="*70)
print()

# CELL 7: Build CRNN Model
def build_crnn(img_h=IMG_HEIGHT, img_w=IMG_WIDTH, n_classes=NUM_CLASSES):
    input_img = layers.Input(shape=(img_h, img_w, 1), name="image")

    x = layers.Rescaling(1.0/255)(input_img)

    # Convolutional layers
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 1))(x)  # Asymmetric pooling

    # Reshape: (batch, h, w, c) → (batch, w, h*c)
    # Register function for Keras 3.x serialization
    @keras.saving.register_keras_serializable(package="custom")
    def conv_to_rnn(inp):
        shape = tf.shape(inp)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        x = tf.transpose(inp, perm=[0, 2, 1, 3])  # (b, w, h, c)
        x = tf.reshape(x, [b, w, h * c])  # (b, w, h*c)
        return x

    x = layers.Lambda(conv_to_rnn, name="to_rnn")(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.2)(x)

    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    # Output logits (no activation)
    logits = layers.Dense(n_classes, name="logits")(x)

    model = keras.Model(inputs=input_img, outputs=logits, name="VTU_CRNN")
    return model

print("BUILDING MODEL")
print("="*70)
base_model = build_crnn()
print("✓ Model built successfully!")
print()
base_model.summary()
print("="*70)
print()

# CELL 8: CTC Layer
class CTCLayer(layers.Layer):
    def __init__(self, name="ctc_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, labels, logits):
        """
        labels: (batch, max_label_len) with -1 as padding
        logits: (batch, time_steps, num_classes)
        """
        batch_len = tf.shape(labels)[0]
        logit_time = tf.shape(logits)[1]

        # Calculate actual label lengths (count non-padding)
        label_len = tf.reduce_sum(tf.cast(tf.not_equal(labels, -1), tf.int32), axis=1)

        # All logit sequences have same length
        logit_len = tf.fill([batch_len], logit_time)

        # Create sparse labels
        nonpad = tf.where(tf.not_equal(labels, -1))
        values = tf.gather_nd(labels, nonpad)
        sparse_shape = tf.cast(tf.shape(labels), tf.int64)
        sparse_labels = tf.SparseTensor(
            indices=tf.cast(nonpad, tf.int64),
            values=tf.cast(values, tf.int32),
            dense_shape=sparse_shape
        )

        # Compute CTC loss
        loss = tf.nn.ctc_loss(
            labels=sparse_labels,
            logits=logits,
            label_length=label_len,
            logit_length=logit_len,
            logits_time_major=False,
            blank_index=BLANK_INDEX
        )

        loss = tf.reduce_mean(loss)
        self.add_loss(loss)

        return logits

# Build training model
label_input = layers.Input(shape=(max_label_len,), dtype=tf.int32, name="label")
logits_with_loss = CTCLayer()(label_input, base_model.output)
training_model = keras.Model(inputs=[base_model.input, label_input], outputs=logits_with_loss)

print("✓ CTC layer configured\n")

# CELL 9: Compile
print("COMPILING MODEL")
print("="*70)
training_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
print("✓ Model compiled (loss added via CTCLayer)")
print("="*70)
print()

# CELL 10: Callbacks
class SaveBestBaseModel(keras.callbacks.Callback):
    def __init__(self, model_to_save, filepath, monitor="val_loss"):
        super().__init__()
        self.model_to_save = model_to_save
        self.filepath = filepath
        self.monitor = monitor
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        current = logs.get(self.monitor)
        if current is None:
            return
        if current < self.best:
            self.best = current
            self.model_to_save.save(self.filepath)
            print(f"\nEpoch {epoch+1}: {self.monitor} improved to {current:.4f}, saving model")

callbacks = [
    SaveBestBaseModel(base_model, os.path.join(MODELS_DIR, 'captcha_model_best.keras')),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOP_PATIENCE,
        verbose=1,
        restore_best_weights=False
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.CSVLogger(os.path.join(MODELS_DIR, 'training_log.csv'))
]

print("✓ Callbacks configured\n")

# CELL 11: Train
print("="*70)
print("STARTING TRAINING")
print("="*70)
print()

history = training_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print()
print("="*70)
print("TRAINING COMPLETE!")
print("="*70)
print()

# CELL 12: Save Final Model
print("SAVING MODELS")
print("="*70)
base_model.save(os.path.join(MODELS_DIR, 'captcha_model.keras'))
print("✓ Final model saved: captcha_model.keras")

with open(os.path.join(MODELS_DIR, 'training_history.json'), 'w') as f:
    hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    json.dump(hist_dict, f, indent=2)
print("✓ Training history saved")
print("="*70)
print()

# CELL 13: Plot Training History
print("PLOTTING TRAINING HISTORY")
print("="*70)

best_epoch = history.history['val_loss'].index(min(history.history['val_loss']))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

min_val_loss = min(history.history['val_loss'])
plt.annotate(
    f'Best: {min_val_loss:.4f}\nEpoch: {best_epoch + 1}',
    xy=(best_epoch, min_val_loss),
    xytext=(best_epoch + 5, min_val_loss + 0.5),
    arrowprops=dict(facecolor='black', arrowstyle='->')
)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (log scale)', fontsize=12)
plt.title('Loss (Log Scale)', fontsize=14, fontweight='bold')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.show()

print("✓ Training history plot saved")
print("="*70)
print()

# CELL 14: Evaluate Model
print("="*70)
print("EVALUATING MODEL")
print("="*70)

# Load best model
# Note: For Keras 3.x, we need to register custom functions before loading
try:
    # Register the custom function for deserialization
    @keras.saving.register_keras_serializable(package="custom")
    def conv_to_rnn(inp):
        shape = tf.shape(inp)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        x = tf.transpose(inp, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [b, w, h * c])
        return x
    
    best = keras.models.load_model(os.path.join(MODELS_DIR, 'captcha_model_best.keras'), compile=False)
    print("✓ Loaded best model")
except Exception as e:
    print(f"⚠️  Best model load failed: {e}")
    print("   Using current base_model instead")
    best = base_model

# Extract validation data
val_images = []
val_labels = []

for batch in val_ds:
    inp, _ = batch
    imgs = inp["image"].numpy()
    lbls = inp["label"].numpy()
    val_images.append(imgs)
    val_labels.append(lbls)

if len(val_images) == 0:
    raise RuntimeError("No validation data found!")

val_images = np.vstack(val_images)
val_labels = np.vstack(val_labels)
print(f"✓ Prepared {len(val_images)} validation samples")

# Get predictions
print("🔄 Making predictions...")
logits = best.predict(val_images, batch_size=BATCH_SIZE, verbose=1)
probs = tf.nn.softmax(logits, axis=-1).numpy()

# Decode with CTC - Keras 3.x compatible
input_len = np.ones(probs.shape[0], dtype=np.int32) * probs.shape[1]
try:
    # Keras 3.x: ctc_decode returns (decoded, log_probabilities)
    decoded_result = keras.backend.ctc_decode(probs, input_length=input_len, greedy=True)
    if isinstance(decoded_result, tuple):
        decoded = decoded_result[0][0].numpy()
    else:
        decoded = decoded_result[0].numpy()
except Exception as e:
    print(f"⚠️  CTC decode error: {e}, using fallback method")
    # Fallback: argmax + collapse
    decoded = []
    for b in range(probs.shape[0]):
        seq = np.argmax(probs[b], axis=-1)
        # Collapse repeats and remove blank
        collapsed = []
        prev = None
        for idx in seq:
            if idx == prev:
                continue
            prev = idx
            if idx != BLANK_INDEX and 0 <= idx < len(CHARACTERS):
                collapsed.append(idx)
        decoded.append(collapsed)
    decoded = np.array(decoded, dtype=object)

def decode_result(decoded_rows):
    texts = []
    for row in decoded_rows:
        text = ""
        for idx in row:
            if 0 <= idx < len(CHARACTERS):
                text += CHARACTERS[int(idx)]
        texts.append(text[:CAPTCHA_LENGTH])
    return texts

pred_texts = decode_result(decoded)

# Decode true labels
true_texts = []
for row in val_labels:
    s = ""
    for idx in row:
        if idx == -1:
            break
        if 0 <= idx < len(CHARACTERS):
            s += CHARACTERS[int(idx)]
    true_texts.append(s)

# Calculate accuracy
total_chars = 0
correct_chars = 0
correct_full = 0

for t, p in zip(true_texts, pred_texts):
    # Pad predictions to match length
    p_padded = p.ljust(len(t))

    if t == p:
        correct_full += 1

    for tc, pc in zip(t, p_padded):
        if tc == pc:
            correct_chars += 1
        total_chars += 1

char_acc = correct_chars / total_chars if total_chars else 0.0
full_acc = correct_full / len(true_texts) if true_texts else 0.0

print()
print("="*70)
print("EVALUATION RESULTS")
print("="*70)
print(f"Character Accuracy: {char_acc*100:.2f}%")
print(f"Full Match Accuracy: {full_acc*100:.2f}%")
print(f"Total Samples: {len(true_texts)}")
print("="*70)
print()

# CELL 15: Sample Predictions
print("SAMPLE PREDICTIONS")
print("="*70)
print()
for i in range(min(20, len(true_texts))):
    status = '✓' if true_texts[i] == pred_texts[i] else '✗'
    print(f"{status} True: {true_texts[i]:8s} | Pred: {pred_texts[i]:8s}")

# Visualize
fig, axes = plt.subplots(4, 5, figsize=(16, 10))
axes = axes.ravel()

for i in range(min(20, len(val_images))):
    axes[i].imshow(val_images[i].squeeze(), cmap='gray')
    is_correct = true_texts[i] == pred_texts[i]
    color = 'green' if is_correct else 'red'
    axes[i].set_title(
        f'True: {true_texts[i]}\nPred: {pred_texts[i]}',
        fontsize=10, color=color, fontweight='bold'
    )
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'predictions_sample.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Sample predictions visualization saved")
print("="*70)
print()

# CELL 16: Save Results
print("SAVING RESULTS")
print("="*70)

# Save detailed results
df = pd.DataFrame({"true": true_texts, "pred": pred_texts})
df["correct"] = df["true"] == df["pred"]
df.to_csv(os.path.join(MODELS_DIR, 'validation_results.csv'), index=False)
print("✓ Validation results saved")

# Save config
config = {
    'img_height': IMG_HEIGHT,
    'img_width': IMG_WIDTH,
    'captcha_length': CAPTCHA_LENGTH,
    'num_classes': NUM_CLASSES,
    'characters': CHARACTERS,
    'batch_size': BATCH_SIZE,
    'epochs_trained': len(history.history['loss']),
    'best_val_loss': float(min(history.history['val_loss'])),
    'char_accuracy': float(char_acc),
    'full_accuracy': float(full_acc)
}

with open(os.path.join(MODELS_DIR, 'model_config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print("✓ Configuration saved")

# Summary
summary = f"""VTU CAPTCHA TRAINING SUMMARY
{'='*70}

PERFORMANCE:
  Character Accuracy: {char_acc*100:.2f}%
  Full Match Accuracy: {full_acc*100:.2f}%
  Best Validation Loss: {min(history.history['val_loss']):.4f}

FILES SAVED:
  1. captcha_model_best.keras ⭐ (use this!)
  2. captcha_model.keras
  3. char_to_int.json
  4. int_to_char.json
  5. model_config.json
  6. training_history.png
  7. predictions_sample.png
  8. validation_results.csv
  9. training_summary.txt

{'='*70}
"""

with open(os.path.join(MODELS_DIR, 'training_summary.txt'), 'w') as f:
    f.write(summary)

print("\n" + summary)

# CELL 17: Create Zip Archive (Local)
print("PREPARING ARCHIVE")
print("="*70)

import shutil
archive_path = os.path.join(os.path.dirname(MODELS_DIR), 'vtu_captcha_trained_models')
try:
    shutil.make_archive(archive_path, 'zip', MODELS_DIR)
    print(f"✓ Zip file created: {archive_path}.zip")
    print(f"  Location: {os.path.abspath(archive_path + '.zip')}")
except Exception as e:
    print(f"⚠️  Could not create zip archive: {e}")
    print("   Model files are still available in:", MODELS_DIR)

print()
print("="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n📁 Model files saved to: {MODELS_DIR}")
print("\n📋 Next steps:")
print("   1. Copy these files to backend/models/:")
print("      - captcha_model_best.keras")
print("      - model_config.json")
print("      - char_to_int.json")
print("      - int_to_char.json")
print("   2. Test the model with backend inference")
print("="*70)