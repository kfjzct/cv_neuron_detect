import matplotlib.pyplot as plt
import processing
import analysis
import numpy as np
import os
from PIL import Image

def load_image(image_path):
    """
    Load an image from the specified path and return the image array along with metadata.

    Parameters:
    image_path : str
        Path to the image file.

    Returns:
    tuple
        Tuple containing the image array, metadata dictionary, and directory of the image.
    """
    meta = {'Name': os.path.basename(image_path).split('.')[0]}  # Extracting the filename as metadata
    directory = os.path.dirname(image_path)  # Extracting the directory path

    # Load and convert the image to an array
    img = Image.open(image_path)
    img = np.array(img)

    # Optionally convert RGB to grayscale if needed
    # if img.ndim == 3:
    #     from skimage.color import rgb2gray
    #     img = rgb2gray(img)

    return img, meta, directory

def main():
    """
    Main function to handle workflow.
    """
    # Define processing parameters
    pixel_density = 3.2
    sigma = 3
    min_samples = 3
    image_path = 'final0.jpg'

    # Load image data
    try:
        neurons, meta, directory = load_image(image_path)
    except FileNotFoundError:
        print(f"Error: The file {image_path} does not exist.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Process the image to find wide clusters
    local_maxi, labels, gauss = processing.wide_clusters(
        neurons,
        sigma=sigma,
        pixel_density=pixel_density,
        min_samples=min_samples
    )

    # Display the Gaussian filtered image
    plt.imshow(gauss, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.title('Gaussian Filtered Image')
    plt.show()

    # Perform segmentation
    ganglion_prop = processing.segmentation(gauss, local_maxi, labels, meta, directory, save=True)
    print(ganglion_prop)

    # Optionally, create a dataframe and save it
    # df, dist = analysis.create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=True)

if __name__ == '__main__':
    main()
