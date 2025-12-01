# 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.saving import register_keras_serializable

#hyper Params
IMAGE_SIZE = 320
PATCH_SIZE = 4 # changed
BATCH_SIZE = 8
NUM_CLASSES = 2
EPOCHS = 10 # changed

# Load dataset using image_dataset_from_directory
dataset = tf.keras.utils.image_dataset_from_directory(
    "Final Drone RF/train",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "Final Drone RF/valid",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)
AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# original DRNN
def ResUnit(x):
    shortcut = x
    conv3x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    conv3x3 = layers.Conv2D(32, (3, 3), padding='same')(conv3x3)
    
    # Add skip connection (not concatenate)
    output = layers.add([conv3x3, shortcut])
    output = layers.ReLU()(output)
    return output
# REsstack
def ResStack(x):
    conv1x1 = layers.Conv2D(32, (1, 1), activation='linear', padding='same')(x)

    output = ResUnit(conv1x1)
    output = ResUnit(output)

    return output

# Define CNN model to extract the features
@register_keras_serializable()
class CNNFeatureExtractor(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):    
        inputs = keras.Input(shape=(320, 320, 3))
        x = layers.Rescaling(1./255)(inputs)  # Normalize input
        x = ResStack(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = ResStack(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = ResStack(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = ResStack(x)
        x = layers.MaxPooling2D((2, 2))(x)
        # output 20x20x32
        self.model = tf.keras.Model(inputs, x)


    def call(self, x):
        return self.model(x)


# Patch and positional encoding layer
@register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
# Embedding the patches with its position + added dense layer
@register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

# MLP head
@register_keras_serializable()
class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout_rate=0.1):
        super().__init__()
        self.hidden_layers = []
        for units in hidden_units:
            # adding non-linear
            self.hidden_layers.append(layers.Dense(units, activation="gelu"))
            # 0.2 drop out
            self.hidden_layers.append(layers.Dropout(dropout_rate))

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x
    
@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, ff_units, dropout_rate=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        key_dim = projection_dim // num_heads
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.add1 = layers.Add()

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(hidden_units=ff_units, dropout_rate=dropout_rate)
        self.add2 = layers.Add()

    def call(self, inputs):
        # Multi-head attention block
        x = self.norm1(inputs)
        attention_output = self.mha(x, x)
        x = self.add1([attention_output, inputs])

        # Feedforward network (MLP)
        y = self.norm2(x)
        y = self.mlp(y)
        return self.add2([y, x])

def create_vit_classifier():
    FEATURE_SIZE = 20 #(20x20)
    num_patches = (FEATURE_SIZE // PATCH_SIZE) ** 2
    projection_dim = 32 
    transformer_units = [projection_dim * 2, projection_dim]  # [128, 64]
    transformer_layers = 2 
    num_heads = 2
    mlp_head_units = [128, 64]

    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    features = CNNFeatureExtractor()(inputs)
    patches = Patches(PATCH_SIZE)(features)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        encoded_patches = TransformerBlock(
            projection_dim=projection_dim,
            num_heads=num_heads,
            ff_units=transformer_units,
            dropout_rate=0.1
        )(encoded_patches)

    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = MLP(hidden_units=mlp_head_units, dropout_rate=0.2)(representation)
    representation = layers.Dropout(0.3)(representation) # remove later

    logits = layers.Dense(NUM_CLASSES)(representation)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

# Compile and train
model = create_vit_classifier()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# Plot and save model architecture
tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

model.summary()
# Train model
history = model.fit(
    dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# Evaluate model
loss, acc = model.evaluate(val_dataset)
print(f"Test Accuracy: {acc:.4f}")

# Save the model
model.save('best_model.keras')


# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
