import streamlit as st
from PIL import Image
import numpy as np
import cv2
from utils import *

class GUI:
    def __init__(self, cnn_model, inv_label_map):
        self.setup_page_config()
        self.apply_custom_css()
        self.cnn_model = None
        self.inv_label_map = inv_label_map

        try:
            self.cnn_model = cnn_model
        except Exception as e:
            st.warning("Model not found: {}".format(e))
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="Image Detection App",
            page_icon="📦",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    
    def apply_custom_css(self):
        st.markdown("""
        <style>
            .main {
                background-color: #f5f7f9;
            }
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .upload-container {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .result-container {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .title {
                color: #2c3e50;
                font-weight: bold;
            }
            .subtitle {
                color: #7f8c8d;
            }
        </style>
        """, unsafe_allow_html=True)

    def process_image(self, img):
        model_input, original_img, preprocessed_dict, segmented_img = test_cnn_on_image_inline(img)
        if model_input is None or original_img is None:
            st.error("Error processing the image. Please try again.")
            return None
        try:
            class_pred, bbox_pred = self.cnn_model.predict(model_input)
            print("Class prediction:", class_pred)
            print("Bounding box prediction:", bbox_pred)
        except Exception as e:
            print("Error during model prediction:", e)
            return
        class_idx = np.argmax(class_pred[0])
        class_name = self.inv_label_map[class_idx]
        bbox = bbox_pred[0]

        # Draw predicted bounding box on the original image
        h, w = original_img.shape[:2]
        x_min = int(bbox[0] * w)
        y_min = int(bbox[1] * h)
        x_max = int(bbox[2] * w)
        y_max = int(bbox[3] * h)

        print(f"Bounding box: ({x_min}, {y_min}), ({x_max}, {y_max})")

        img_copy = original_img.copy()
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(img_copy, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        return img_rgb, class_name, preprocessed_dict, segmented_img

    
    def predict(self, image):
        """
        Predict class using the loaded model or fall back to mock predictions
        """
        pass

    def render_sidebar(self):
        st.sidebar.title("About")
        st.sidebar.info(
            "This application detects and classifies objects in uploaded images. "
            "It uses computer vision techniques to segment the image and identify "
            "objects with bounding boxes."
        )
        st.sidebar.title("Instructions")
        st.sidebar.markdown(
            "1. Upload an image containing an object\n"
            "2. Wait for the segmentation process\n"
            "3. View the detected object with bounding box\n"
            "4. See the predicted class"
        )
    
    def render_header(self):
        st.markdown("<h1 class='title'>📦 Image Detection App</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Upload a photo of an object to detect and classify it</p>", unsafe_allow_html=True)
    
    def render_uploader(self):
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)
        return uploaded_file
    
    
    def run(self):
        self.render_header()
        self.render_sidebar()
        
        uploaded_file = self.render_uploader()
        
        if uploaded_file is not None:
            
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            
            # Open the image file
            image = Image.open(uploaded_file)
            # path = uploaded_file.name
            img = np.array(image)
        
            
            # Display original image
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Process the image
            output, class_name, preprocessed_dict, segmented_img = self.process_image(img)
            
            # Display results
            if output is not None:
                with col2:
                    st.subheader("Detection Result")
                    st.image(output, use_container_width=True)
                    fig, axes = plt.subplots(1, 3, figsize=(5, 5))
                    axes[0].imshow(preprocessed_dict["resized"])
                    axes[0].set_title("Resized")
                    axes[0].axis("off")
                    axes[1].imshow(segmented_img)
                    axes[1].set_title("Segmented")
                    axes[1].axis("off")
                    axes[2].imshow(preprocessed_dict["contrast"])
                    axes[2].set_title("Contrast")
                    axes[2].axis("off")
                    st.pyplot(fig)
                
                # Display prediction
                st.success(f"✅ Detection complete!")
                st.metric("Predicted Class", class_name)
            else:
                st.error("No fruit detected in the image. Please try with another image.")
            
            st.markdown("</div>", unsafe_allow_html=True)

