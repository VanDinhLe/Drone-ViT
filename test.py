import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, 
    confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay,
    precision_recall_curve, auc, roc_curve, precision_recall_curve
)

from tensorflow.keras.saving import register_keras_serializable

IMAGE_SIZE = 320
PATCH_SIZE = 4 # changed
BATCH_SIZE = 8
NUM_CLASSES = 2
EPOCHS = 10 # changed
# load dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "Final Drone RF/test",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle = False
)
class_names = test_dataset.class_names
# loading classes for loading all the weights
def ResUnit(x):
    shortcut = x
    conv3x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    conv3x3 = layers.Conv2D(32, (3, 3), padding='same')(conv3x3)
    
    # Add skip connection (not concatenate)
    output = layers.add([conv3x3, shortcut])
    output = layers.ReLU()(output)
    return output
    
def ResStack(x):
    conv1x1 = layers.Conv2D(32, (1, 1), activation='linear', padding='same')(x)

    output = ResUnit(conv1x1)
    output = ResUnit(output)

    return output

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

        self.model = tf.keras.Model(inputs, x)


    def call(self, x):
        return self.model(x)


# Patch and positional encoding layer
@register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
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
@register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

@register_keras_serializable()
class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(layers.Dense(units, activation="gelu"))
            self.hidden_layers.append(layers.Dropout(dropout_rate))

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x
    
@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, ff_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
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
# load the model
model = tf.keras.models.load_model("best_model.keras", custom_objects={
    "CNNFeatureExtractor": CNNFeatureExtractor,
    "Patches": Patches,
    "PatchEncoder": PatchEncoder,
    "TransformerBlock": TransformerBlock,
    "MLP": MLP
})

# evaluate the model on data set
loss, acc = model.evaluate(test_dataset)
print(f"Test Accuracy (Keras): {acc:.4f}")

# set true labels and  predictions
y_true = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)
y_pred_logits = model.predict(test_dataset)
y_pred = np.argmax(y_pred_logits, axis=1)
y_prob = tf.nn.softmax(y_pred_logits, axis=1).numpy()
y_score = y_prob[:, 1]

#  print the metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_score)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

#  plot the confusion matrix 
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Test Set)")
plt.show()

# plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc_manual = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc_manual:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) - Test Set")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#plot PR curve
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_score)
avg_precision = average_precision_score(y_true, y_score)

plt.figure(figsize=(7, 6))
plt.plot(recall_vals, precision_vals, color="purple", lw=2,
         label=f"PR curve (AP = {avg_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Test Set)")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()
