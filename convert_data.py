import os
import sys
import numpy as np
from matplotlib.pyplot import imread
from scipy.ndimage import zoom

# Need to check if sampling data is correct, if not then we need to fix it
def main():
    # Ensure correct number of command-line arguments
    if len(sys.argv) != 6:
        print("Usage: python3 convert_data.py <image_folder> <h> <w> <c> <npz_output_file>")
        sys.exit(1)

    # Parse command-line arguments
    image_folder = sys.argv[1]
    h, w, c = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    output_file = sys.argv[5]

    # Validate h, w, c parameters (should always be 28, 28, and 1)
    if h != 28 or w != 28 or c != 1:
        print("Error: h, w, c must be 28, 28, and 1.")
        sys.exit(1)

    # Initialize lists for image data (X) and labels (T)
    X = []
    T = []

    # Loop through all files in the image folder
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)

        # Skip non-PNG files
        if not file_name.endswith(".png"):
            continue

        try:
            # Load image using matplotlib's imread (returns NumPy array)
            img = imread(file_path)

            # Resize image to (28x28)
            zoom_factor = (h / img.shape[0], w / img.shape[1])
            img_resized = zoom(img, zoom_factor)

            # Normalize pixel values to [0, 1]
            img_array = img_resized / 255.0

            # Add channel dimension if c == 1
            img_array = np.expand_dims(img_array, axis=-1)

            X.append(img_array)

            # Extract label from filename (assumes format like "label-image.png")
            label = int(file_name.split('-')[0])
            T.append(label)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    # Convert lists to NumPy arrays
    X = np.array(X, dtype=np.float32)
    T = np.array(T, dtype=np.int32)

    # Save X and T into an .npz file
    np.savez(output_file, X=X, T=T)
    print(f"Data saved to {output_file}.")
    print(f"Shape of X: {X.shape}, Shape of T: {T.shape}")

if __name__ == "__main__":
    main()