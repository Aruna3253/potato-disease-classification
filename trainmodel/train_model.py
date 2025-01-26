import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.saving import register_keras_serializable
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Configuration
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
CHANNELS = 3
EPOCHS = 50

@register_keras_serializable(package='CustomModels')
class CustomConvLayer(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', **kwargs):
        super(CustomConvLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        # Custom weight initialization
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
            initializer='he_normal',
            regularizer=tf.keras.regularizers.l2(0.001)
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros'
        )
        super(CustomConvLayer, self).build(input_shape)

    def call(self, inputs):
        # Custom convolution implementation
        conv_output = tf.nn.conv2d(
            inputs, 
            self.kernel, 
            strides=[1, *self.strides, 1], 
            padding=self.padding.upper()
        )
        return tf.nn.relu(conv_output + self.bias)

    def get_config(self):
        config = super(CustomConvLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config

@register_keras_serializable(package='CustomModels')
class CustomCNN(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(CustomCNN, self).__init__(**kwargs)
        
        # Custom data augmentation layers
        self.augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1)
        ])
        
        # Custom convolutional layers
        self.conv1 = CustomConvLayer(32, 3)
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = layers.Dropout(0.25)
        
        self.conv2 = CustomConvLayer(64, 3)
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = layers.Dropout(0.35)
        
        self.conv3 = CustomConvLayer(128, 3)
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout3 = layers.Dropout(0.45)
        
        # Flatten and dense layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu', 
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout4 = layers.Dropout(0.5)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Apply augmentation only during training
        x = self.augmentation(inputs, training=training) if training else inputs
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout3(x, training=training)
        
        # Dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout4(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        config = super(CustomCNN, self).get_config()
        config.update({
            'num_classes': self.output_layer.units
        })
        return config

def prepare_dataset(path):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset='validation', 
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = dataset.class_names
    
    # Normalize and prefetch
    train_ds = dataset.map(
        lambda x, y: (x / 255.0, y), 
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_dataset.map(
        lambda x, y: (x / 255.0, y), 
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

def train_model(train_ds, val_ds, class_names):
    # Extract labels for class weights
    train_labels = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
    
    # Compute balanced class weights
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_labels), 
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights))

    # Model setup
    model = CustomCNN(num_classes=len(class_names))
    
    # Optimizer with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5
        )
    ]

    # Model training
    history = model.fit(
        train_ds, 
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return model, history

if __name__ == "__main__":
    # Dataset path
    DATASET_PATH = "training/Plantvillage"
    
    # Load and prepare dataset
    train_ds, val_ds, class_names = prepare_dataset(DATASET_PATH)
    
    # Save class names
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # Train model
    model, history = train_model(train_ds, val_ds, class_names)

    # Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save model
    model.save("custom_plant_disease_model.keras")