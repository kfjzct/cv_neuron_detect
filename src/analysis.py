import pandas as pd
import numpy as np

def create_dataframe(ganglion_prop, labels, local_maxi, meta, directory, save=False):
    """
    Creates a DataFrame from ganglion properties and optionally saves it to a CSV file.

    Parameters:
    ganglion_prop : list
        List of properties for each ganglion detected in the image.
    labels : np.array
        Array of label data corresponding to different ganglions.
    local_maxi : list
        List of local maximum points detected in the image.
    meta : dict
        Metadata dictionary containing necessary information like PhysicalSizeX and Name.
    directory : str
        Directory path where the CSV file will be saved if save is True.
    save : bool
        Flag to determine whether to save the DataFrame to a CSV file.

    Returns:
    tuple
        A tuple containing the DataFrame and the histogram plot object.
    """
    if not ganglion_prop:
        # Initialize DataFrame with zeros if no properties are provided
        column_names = ['ganglion', 'Nbr of neurons', "surface ganglion", "major axis length", "minor axis length", "orientation"]
        initial_data = np.zeros((1, 6))
        df = pd.DataFrame(initial_data, columns=column_names)
    else:
        results = []
        intergang = 0
        label_new = np.copy(labels)
        label_new[labels == 0] = max(labels) + 1

        # Collect data for each ganglion
        for prop in ganglion_prop:
            label = prop.label
            neurons_in_label = np.where(label_new == label)[0]
            number_of_neurons = len(neurons_in_label)
            results.append((label, number_of_neurons))
            intergang += number_of_neurons

        results = np.asarray(results)
        df = pd.DataFrame(results, columns=['ganglion', 'Nbr of neurons'])

        # Assign additional properties to DataFrame
        df["surface area"] = [prop.area for prop in ganglion_prop]
        df["major axis length"] = [prop.major_axis_length for prop in ganglion_prop]
        df["minor axis length"] = [prop.minor_axis_length for prop in ganglion_prop]
        df["orientation"] = [prop.orientation for prop in ganglion_prop]

        # Convert lengths based on physical size from metadata
        df['major axis length'] *= meta['PhysicalSizeX']
        df['minor axis length'] *= meta['PhysicalSizeX']

        # Rename columns for clarity
        df = df.rename(columns={
            "surface area": "surface area in um2",
            "major axis length": "major axis length in um",
            "minor axis length": "minor axis length in um"
        })

        # Create histogram for the number of neurons
        histogram_plot = df.hist(column='Nbr of neurons', bins=20)

        # Adjust DataFrame to include summary data
        df.replace({"ganglion": 0}, "total in field", inplace=True)
        df.loc[-1] = np.nan
        df.index = df.index + 1
        df.sort_index(inplace=True)

        df.at[0, 'ganglion'] = "total in field"
        total_neuron_count = len(local_maxi)
        df.at[0, "Nbr of neurons"] = total_neuron_count

        extragang = total_neuron_count - intergang
        df = df.append({'ganglion': "intra-ganglionic total", 'Nbr of neurons': intergang}, ignore_index=True)
        df = df.append({'ganglion': "extra-ganglionic total", 'Nbr of neurons': extragang}, ignore_index=True)

    if save:
        # Save DataFrame to a CSV file
        file_path = os.path.join(directory, f"{meta['Name']}.csv")
        try:
            df.to_csv(file_path)
        except FileNotFoundError:
            fallback_path = f"{meta['Name']}.csv"
            df.to_csv(fallback_path)

    return df, histogram_plot
