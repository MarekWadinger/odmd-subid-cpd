import os

import numpy as np
import requests


def load_dateset(file_path, url, save: bool = False):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the data from the file into a numpy array
        return np.loadtxt(file_path)
    else:
        # If the file does not exist, download the data
        response = requests.get(url)

        if response.status_code == 200:
            lines = response.text.split("\n")
            data = np.array(
                [float(line.strip()) for line in lines if line.strip()]
            )
            if save:
                # Check if the directory exists, if not create it
                directory = os.path.dirname(file_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                print(f"Saving dataset to {file_path}")
                # Save the data to file_path
                np.savetxt(file_path, data)
            return data
        else:
            raise ValueError(
                f"Error {response.status_code} while downloading the nprs44. "
                f"Check connection or download and store manually from {url} "
                f"to {file_path}"
            )


def load_nprs43():
    return load_dateset(
        "data/nprs/nprs43.txt",
        "https://www.cs.ucr.edu/~eamonn/discords/nprs43.txt",
        save=True,
    )


def load_nprs44():
    return load_dateset(
        "data/nprs/nprs44.txt",
        "https://www.cs.ucr.edu/~eamonn/discords/nprs44.txt",
        save=True,
    )
