import numpy as np
import cv2
import glob

def preprocess_image(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Flatten the image into a 1D array
    flattened_image = image.flatten()

    # Normalize the pixel values between 0 and 1
    normalized_image = flattened_image / 255.0
    return normalized_image

def preprocess_folder(folder_path):
    # Get a list of all image files in the folder
    image_files = glob.glob(folder_path + '/*.png')
    
    # Preprocess each image and store the results in a list
    preprocessed_images = []
    labels = []
    for image_file in image_files:
        preprocessed_image = preprocess_image(image_file)
        preprocessed_images.append(preprocessed_image)
        labels.append(int(image_file[-5]))
    # Convert the list of preprocessed images into a NumPy array
    preprocessed_images = np.array(preprocessed_images)
    labels = np.array(labels)
    return preprocessed_images,labels

