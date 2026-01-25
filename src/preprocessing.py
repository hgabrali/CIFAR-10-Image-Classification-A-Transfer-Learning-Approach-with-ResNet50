import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input

def prepare_data(train_images, train_labels, test_images, test_labels):
    """
    Standardizes and encodes CIFAR-10 data for ResNet50 compatibility.
    
    Steps:
    1. Convert pixel values to float32.
    2. Apply ResNet50 specific preprocessing (scaling/centering).
    3. One-hot encode the categorical labels.
    """

    # 1. Cast images to float32 for precision during mathematical operations
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # 2. ResNet50 specific preprocessing
    # This function scales pixels and deals with RGB mean subtraction 
    # as required by the pre-trained ImageNet weights.
    x_train = preprocess_input(train_images)
    x_test = preprocess_input(test_images)

    # 3. One-hot encoding for the 10 mutually exclusive classes
    # Converts integer labels (e.g., 3) into a binary vector ([0,0,0,1,0...])
    y_train = to_categorical(train_labels, 10)
    y_test = to_categorical(test_labels, 10)

    return x_train, y_train, x_test, y_test