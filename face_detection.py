import os
import cv2
import argparse
from facenet_pytorch import MTCNN
import torch

# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description='Face detection using PyTorch-based MTCNN.')
    
    # Input directory argument
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing input images.')
    
    # Output directory argument
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the directory where cropped faces will be saved.')
    
    return parser.parse_args()

# Main function
def main():
    # Parse command-line arguments
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize MTCNN face detector from facenet-pytorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    # Loop through all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            img_path = os.path.join(input_dir, filename)
            
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read {filename}. Skipping.")
                continue

            img_height, img_width = img.shape[:2]
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MTCNN works on RGB images

            # Detect faces using MTCNN
            boxes, _ = mtcnn.detect(rgb_img)

            # If faces are detected
            if boxes is not None:
                valid_faces = []

                # Check each face to see if it is larger than 100x100
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]  # Get bounding box as integers

                    # Calculate the width and height of the detected face
                    face_width = x2 - x1
                    face_height = y2 - y1

                    # Only consider faces that are larger than 100x100
                    if face_width >= 100 and face_height >= 100:
                        valid_faces.append((x1, y1, x2, y2))

                # If there are multiple valid faces, skip the image
                if len(valid_faces) > 1:
                    print(f"Skipped {filename} because multiple valid faces were found.")
                    continue

                # If exactly one valid face is found, process and save it
                if len(valid_faces) == 1:
                    x1, y1, x2, y2 = valid_faces[0]

                    # Clip the coordinates to ensure they are within the image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)

                    # Crop the face from the image
                    face_crop = img[y1:y2, x1:x2]

                    # Only save if the face crop has non-zero dimensions
                    if face_crop.size > 0:
                        # Save the cropped face to the output directory as a PNG
                        face_filename = f"{os.path.splitext(filename)[0]}_face.png"
                        face_path = os.path.join(output_dir, face_filename)
                        cv2.imwrite(face_path, face_crop)  # Save the cropped face as PNG
                        print(f"Saved cropped face from {filename}.")
                    else:
                        print(f"Failed to crop a valid face for {filename}. Skipping.")
                else:
                    print(f"No valid faces found in {filename}.")
            else:
                print(f"No faces found in {filename}.")
        else:
            print(f"Skipped non-image file {filename}.")

if __name__ == "__main__":
    main()
