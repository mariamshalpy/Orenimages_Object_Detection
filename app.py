from src.GUI import GUI
from Utils.utils import Get_Model_cnn

cnn_model_path = "custom_cnn_model.h5"
label_map = {'apple': 0, 'banana': 1, 'bicycle': 2, 'car': 3, 'chair': 4, 'dog': 5, 'person': 6}
inv_label_map = {v: k for k, v in label_map.items()}

# Load the models
cnn_model = Get_Model_cnn(cnn_model_path)
app = GUI(cnn_model=cnn_model, inv_label_map=inv_label_map)

if __name__ == "__main__":
    app.run()


