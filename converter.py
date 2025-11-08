import tensorflow as tf

def export_tflite(model, filename="edge_model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(filename, "wb") as f:
        f.write(tflite_model)
    print(f" Model saved to {filename}")

# Load the trained model
model = tf.keras.models.load_model("edge_model.h5")

# Export to TFLite
export_tflite(model)
