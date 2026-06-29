import numpy as np
import cv2
import tensorflow as tf
from datasets import load_dataset

def get_dataset_pipeline(split_name='train', batch_size=32):
    # using 'test' because the dataset doesnt have a train
    dataset = load_dataset("nguyenkhoa/antispoofing-3", split=split_name, streaming=True)

    def preprocess(example):
        label = 0 if example['label'] == 0 else 1
        
        # Convert and resize
        image = np.array(example['cropped_image'].convert('RGB'))
        image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype("float32") / 255.0
        return image, label

    def gen():
        for ex in dataset:
            if ex.get('cropped_image') is not None:
                yield preprocess(ex)

    output_sig = (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    return tf.data.Dataset.from_generator(
        gen, 
        output_signature=output_sig
    ).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = get_dataset_pipeline(split_name='train')
val_ds = get_dataset_pipeline(split_name='eval')