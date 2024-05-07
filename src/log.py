import os

def log_file(directory, meta, **kwargs):
    """
    Creates a log file in the specified directory with information about image processing parameters.

    Parameters:
    directory : str
        The directory where the log file will be saved.
    meta : dict
        Metadata dictionary containing at least a "Name" key for the filename.
    kwargs : dict
        Additional keyword arguments for image processing parameters:
        - sigma : float
            Sigma value used for Gaussian blurring.
        - min_distance : int
            Minimum distance between detected peaks.
        - eps : float
            Epsilon value, the maximum distance between two samples for them to be considered as in the same neighborhood.
        - min_samples : int
            Minimum number of samples in a neighborhood for a point to be considered as a core point.
    """

    # Ensure all required parameters are provided
    required_params = ['sigma', 'min_distance', 'eps', 'min_samples']
    for param in required_params:
        if param not in kwargs:
            raise ValueError(f"Missing required parameter: {param}")

    # Construct the filename from the directory and metadata
    filename = os.path.join(directory, meta.get("Name", "log.txt"))

    # Write the log information to the file
    try:
        with open(filename, 'w') as file:
            file.write(
                f"Size of the Sigma used for the blurring of the image: {kwargs['sigma']}\n"
                f"Minimum distance between 2 peaks (nucleus): {kwargs['min_distance']}\n"
                f"Minimum distance between cells within a cluster: {kwargs['eps']}\n"
                f"Minimum amount of cells to form a cluster: {kwargs['min_samples']}\n"
            )
    except IOError as e:
        print(f"An error occurred while writing to the file: {filename}. Error: {e}")

if __name__ == "__main__":
    # Example usage
    directory = 'path/to/directory'
    meta = {'Name': 'example_log.txt'}
    parameters = {
        'sigma': 2.0,
        'min_distance': 25,
        'eps': 1.5,
        'min_samples': 10
    }

    log_file(directory, meta, **parameters)
