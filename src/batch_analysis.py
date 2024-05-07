import javabridge
import bioformats
import io as io
import processing as processing
import analysis as analysis
import os
import pandas as pd


def batch_analysis(directory_path, output_directory, sigma_value, pixel_density_value, minimum_samples, **kwargs):
    """
    directory_path : str
        Path to the directory containing image files.
    output_directory : str
        Path where output files should be saved.
    sigma_value : float
        Sigma value for Gaussian blurring in image processing.
    pixel_density_value : float
        Pixel density threshold for image analysis.
    minimum_samples : int
        Minimum number of samples for clustering algorithms.
    kwargs : dict
        Additional keyword arguments, can include:
        - imageformat: str, specifying the file extension of images to process.
    """

    # Retrieve the image format from the kwargs; set default to '.tif' if not provided
    image_format = kwargs.get('imageformat', '.tif')

    # List all files in the directory that match the image format
    image_file_list = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if
                       file.endswith(image_format)]

    # Process each file in the list
    for image_file in image_file_list:

        print(f"Processing file: {image_file}")

        # Initialize the Java VM for bioformats package
        javabridge.start_vm(class_path=bioformats.JARS)

        # Extract metadata from the image file
        _, number_of_series = io._metadata(image_file)

        for series_index in range(number_of_series):
            # Load the TIFF image as Maximum Intensity Projection (MIP), along with directory and metadata
            mip_image, image_directory, image_metadata = io.load_TIFF(image_file, output_directory, serie=series_index)

            # Perform clustering to identify wide clusters
            local_maxima, label_image, gaussian_filtered = processing.wide_clusters(
                mip_image,
                sigma=sigma_value,
                pixel_density=pixel_density_value,
                min_samples=minimum_samples,
                plot=False
            )
            del mip_image  # Free up memory by deleting the MIP image

            # Perform segmentation based on the Gaussian filtered image and local maxima
            ganglion_properties = processing.segmentation(
                gaussian_filtered,
                local_maxima,
                label_image,
                image_metadata,
                image_directory,
                save=True
            )
            del gaussian_filtered  # Free up memory by deleting the Gaussian filtered image

            # Create and save the dataframe with the analysis results
            analysis.create_dataframe(ganglion_properties, label_image, local_maxima, image_metadata, image_directory,
                                      save=True)

        # Shutdown Java VM to free up resources
        javabridge.kill_vm()
