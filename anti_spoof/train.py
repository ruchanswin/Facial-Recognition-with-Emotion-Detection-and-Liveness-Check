from model import build_liveness_model
from dataset import get_dataset_pipeline
import tensorflow as tf

# Load Data
train_ds = get_dataset_pipeline(split_name='train', batch_size=32)
val_ds = get_dataset_pipeline(split_name='eval', batch_size=32)

# Build and Compile
model = build_liveness_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Train
print("Starting training on AxonData...")
model.fit(train_ds, validation_data=val_ds, steps_per_epoch=500, epochs=10, validation_steps=50)

# Save for GUI integration
model.save('saved_models/antispoof_v3.h5')