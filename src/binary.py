import cv2
import numpy as np


def convert_to_binary(image):
    """
    Converts an image to a binary format using Otsu's thresholding after converting it to grayscale.

    Parameters:
    image : np.array
        Original image in BGR format.

    Returns:
    binary_image : np.array
        Binary image after applying Otsu's threshold.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding after Gaussian filtering to binarize the image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def main():
    # Load an image from the disk
    image_path = 'final0.jpg'
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Image not found at {image_path}")
        return

    # Convert image to binary
    binary_image = convert_to_binary(original_image)

    # Find connected components in the binary image
    num_labels, labels_image = cv2.connectedComponents(binary_image)

    # Define minimum area in pixels to keep
    minimum_area = 75
    new_image = np.zeros_like(binary_image)

    # Filter small components based on the minimum area
    for label in range(1, num_labels):
        if np.sum(labels_image == label) > minimum_area:
            new_image[labels_image == label] = 255

    # Display the original and cleaned binary images
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Cleaned Binary Image', new_image)

    # Save the cleaned image to disk
    output_image_path = "new.jpg"
    cv2.imwrite(output_image_path, new_image)

    # Wait for a key press and then terminate all OpenCV windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
