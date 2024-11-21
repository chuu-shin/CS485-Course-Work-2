import os
import cv2
import numpy as np

# Path to the dataset
data_path = './assets/train/'

# Feature extractor (SIFT)
sift = cv2.SIFT_create(contrastThreshold=0.01)

def extract_descriptors_from_image(image_path, descriptors_per_image=DESCRIPTORS_PER_IMAGE):
    """
    Extract exactly a specified number of descriptors from an image using SIFT.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or invalid: {image_path}")
    
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is None or len(descriptors) == 0:
        raise ValueError(f"No descriptors found in image: {image_path}")
    
    selected_descriptors = descriptors
    
    return selected_descriptors

def collect_descriptors_with_labels(data_path, descriptors_per_image=DESCRIPTORS_PER_IMAGE):
    """
    Collect descriptors and their corresponding labels.
    """
    all_descriptors = []
    all_labels = []
    
    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    num_des_tot = 0
    num_des_class_dist = []
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(data_path, class_name)
        print(f"Processing class: {class_name}")
        
        num_des_class = 0
        for image_name in os.listdir(class_folder):
            if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(class_folder, image_name)
                try:
                    # Extract descriptors for this image
                    descriptors = extract_descriptors_from_image(image_path, descriptors_per_image)
                    
                    # Append descriptors and corresponding labels
                    all_descriptors.append(descriptors)
                    all_labels.append(i)
                    num_des_class += len(descriptors)
                except ValueError as e:
                    print(f"Error processing {image_name}: {e}")
        num_des_tot += num_des_class
        num_des_class_dist.append(num_des_class)
    print(num_des_tot)
    
    # Convert to NumPy arrays
    all_descriptors = np.array(all_descriptors, dtype=object)
    all_labels = np.array(all_labels)
    
    return all_descriptors, all_labels

# Main function
if __name__ == "__main__":
    # Extract descriptors and labels
    descriptors, labels = collect_descriptors_with_labels(data_path)
    print("Descriptor extraction completed.")
    
    # Save the descriptors and labels
    np.save('descriptors.npy', descriptors)
    np.save('labels.npy', labels)
