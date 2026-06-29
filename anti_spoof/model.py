import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model

def build_liveness_model(input_shape=(224, 224, 3)):
    # Base model with ImageNet weights
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", 
        include_top=False, 
        input_tensor=Input(shape=input_shape)
    )
    
    for layer in base_model.layers:
        layer.trainable = False
        
    # Custom head for binary classification
    head = base_model.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(1, activation="sigmoid")(head) 
    
    return Model(inputs=base_model.input, outputs=head)