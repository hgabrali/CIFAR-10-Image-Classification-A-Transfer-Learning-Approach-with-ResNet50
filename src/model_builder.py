import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def build_transfer_learning_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Initializes a ResNet50 base model with ImageNet weights and appends a custom head.
    
    Architecture Steps:
    1. Load ResNet50 without the top classification layer.
    2. Freeze base layers to preserve pre-trained knowledge.
    3. Add a GlobalAveragePooling layer to reduce feature dimensionality.
    4. Attach a custom classifier 'head' with two dense layers.
    """

    # 1. Initialize the Base Model
    # include_top=False removes the final 1000-class dense layer of ResNet50
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          input_shape=input_shape)

    # 2. Freezing the Base Model
    # This prevents gradients from updating the pre-trained weights during initial training
    base_model.trainable = False

    # 3. Building the Custom 'Head'
    # We follow the recommendation of using two hidden layers (128 and 64 neurons)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax') # Output layer for 10 classes
    ])

    return model
