from src.UI import GUI
import os
import time
import tempfile
cnn_model_path = "models/custom_cnn_model.h5"
label_map = {'apple': 0, 'banana': 1, 'bicycle': 2, 'car': 3, 'chair': 4, 'dog': 5, 'person': 6}
inv_label_map = {v: k for k, v in label_map.items()}

# Load the models
app = GUI(cnn_model_path, inv_label_map=inv_label_map)

if __name__ == "__main__":
    app.run()

