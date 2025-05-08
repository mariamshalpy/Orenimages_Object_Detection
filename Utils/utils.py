import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import cv2


def preprocess_image(image, target_size=(256, 256)):
    """
    Perform preprocessing and image enhancement.
    Returns original, grayscale, HSV, and binary images for further processing.
    """
    # Resize
    image_resized = cv2.resize(image, target_size)

    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0

    # Convert to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Denoise using Gaussian blur
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    # Thresholding using Otsu
    _, binary_otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    morph_open = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Convert to HSV for possible later use
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)

    return {
        "resized": image_resized,
        "normalized": image_normalized,
        "gray": gray,
        "contrast": contrast,
        "binary": morph_close,
        "hsv": hsv
    }


def watershed_segmentation2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Background and foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    img_color = img.copy()
    markers = cv2.watershed(img_color, markers)

    # Extract binary masks for each object
    masks = []
    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue  # Skip background/border
        mask = np.uint8(markers == marker_id)
        masks.append(mask)

    return masks  # List of binary masks for each segmented object



# Watershed Segmentation
def watershed_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Background and foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    img_color = img.copy()
    markers = cv2.watershed(img_color, markers)

    # Draw red boundaries on the original image
    img_color[markers == -1] = [0, 0, 255]

    return img_color

def segment_image(image_dict):
    """
    Segment an image using the Watershed method.
    """
    return watershed_segmentation(image_dict["resized"])

def extract_rois(img, roi_size=(64, 64)):
    height, width = img.shape[:2]
    print(f"Image size: {height}x{width}") 
    if height < roi_size[0] or width < roi_size[1]:
        print("Image is too small for the given ROI size. Resizing the image.")
        img = cv2.resize(img, (max(roi_size[1], width), max(roi_size[0], height)))
        height, width = img.shape[:2] 
    # Adjust step size based on ROI size
    step_size = max(roi_size[0], roi_size[1])
    rois = []
    bboxes = []
    for y in range(0, height - roi_size[0] + 1, step_size):
        for x in range(0, width - roi_size[1] + 1, step_size):
            roi = img[y:y + roi_size[0], x:x + roi_size[1]]
            rois.append(roi)
            bboxes.append([x, y, x + roi_size[1], y + roi_size[0]])  # The bounding box for this ROI

    roi_images = [cv2.resize(roi, roi_size) for roi in rois]
    roi_images = np.array(roi_images)  # Prepare input for the model
    roi_images = roi_images.astype("float32") / 255.0
    return roi_images, bboxes

def scale_bboxes_to_full_image(bboxes, original_size, roi_size=(64, 64)):
    scale_x = original_size[1] / roi_size[1]
    scale_y = original_size[0] / roi_size[0]
    
    scaled_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min_scaled = x_min * scale_x
        y_min_scaled = y_min * scale_y
        x_max_scaled = x_max * scale_x
        y_max_scaled = y_max * scale_y
        scaled_bboxes.append([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled])
    
    return scaled_bboxes

def process_image_for_prediction(original_img):
    # Read and preprocess image
    # original_img = cv2.imread(img_path)
    if original_img is None:
      raise ValueError(f"Image could not be loaded. Check the path.")
    print("Original image shape:", original_img.shape)

    preprocessed_dict = preprocess_image(original_img)
    segmented_img = segment_image(preprocessed_dict)
    print("Segmented image shape:", segmented_img.shape)

    # Resize and normalize for model input
    model_input = cv2.resize(segmented_img, (64, 64)).astype("float32") / 255.0
    model_input = np.expand_dims(model_input, axis=0)
    print("Model input shape:", model_input.shape)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    return model_input, original_img, preprocessed_dict, segmented_img


def Get_Model_cnn(model_path):
    """
    Load the pre-trained model for fruit detection.
    """
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    return model

def Save_Model_cnn(model, save_path):
    """
    Save the trained model to a file.
    """
    model.save(save_path)
    return save_path

