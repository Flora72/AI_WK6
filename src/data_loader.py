import tensorflow as tf
import numpy as np

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    class_names = ['plastic', 'paper', 'glass']
    
    x_train = x_train[:3000] / 255.0
    y_train = y_train[:3000] % 3
    
    x_test = x_test[:500] / 255.0
    y_test = y_test[:500] % 3
    
    y_train = tf.keras.utils.to_categorical(y_train, 3)
    y_test = tf.keras.utils.to_categorical(y_test, 3)
    
    return (x_train, y_train), (x_test, y_test), class_names