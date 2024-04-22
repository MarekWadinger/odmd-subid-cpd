import os

import numpy as np
import pandas as pd
import requests


def load_dateset(file_path, url, save: bool = False):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the data from the file into a numpy array
        return np.loadtxt(file_path)

    # If the file does not exist, download the data
    response = requests.get(url)

    if response.status_code == 200:
        lines = response.text.split("\n")
        data = np.array(
            [float(line.strip()) for line in lines if line.strip()]
        )
    else:
        raise ValueError(
            f"Error {response.status_code} while downloading the nprs44. "
            f"Check connection or download and store manually from {url} "
            f"to {file_path}"
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


def load_nprs43() -> np.ndarray:
    return load_dateset(
        "data/nprs/nprs43.txt",
        "https://www.cs.ucr.edu/~eamonn/discords/nprs43.txt",
        save=True,
    )


def load_nprs44() -> np.ndarray:
    return load_dateset(
        "data/nprs/nprs44.txt",
        "https://www.cs.ucr.edu/~eamonn/discords/nprs44.txt",
        save=True,
    )


def load_cats(file_path: str = "data/cats/data.csv") -> pd.DataFrame:
    url = "https://zenodo.org/records/7646897/files/data.parquet"

    if os.path.exists(file_path):
        # Read the data from the file into a numpy array
        return pd.read_csv(file_path, index_col=0)

    def download_and_read_parquet_with_progress(url):
        """
        Download a Parquet file from the given URL, save it to memory, and read
        it into a pandas DataFrame, while printing the download progress.

        Parameters:
            url (str): The URL of the Parquet file to download and read.

        Returns:
            pandas DataFrame: The DataFrame containing the data from the Parquet file.
        """
        from io import BytesIO

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        bytes_downloaded = 0

        buffer = BytesIO()
        for data in response.iter_content(chunk_size=1048576):
            buffer.write(data)
            bytes_downloaded += len(data)
            progress = bytes_downloaded / total_size * 100
            print(
                f"Downloaded {bytes_downloaded}/{total_size} bytes ({progress:.2f}%)\r",
                end="",
            )

        # Reset buffer position to the beginning before reading
        buffer.seek(0)

        # Read the Parquet file from the buffer into a pandas DataFrame
        df = pd.read_parquet(buffer)
        return df

    # Example usage
    df = download_and_read_parquet_with_progress(url)

    # Check if the directory exists, if not create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f"Saving dataset to {file_path}")
    # Save the data to file_path
    df.to_csv(file_path)

    return df


def load_skab(file_path: str = "data/skab") -> dict[str, list[pd.DataFrame]]:
    from urllib.parse import urlparse

    url = "https://api.github.com/repos/waico/SKAB/contents/data"

    if not os.path.exists(file_path):

        def download_csv_from_git(url, save_path, add_base: bool = True):
            # Parse the URL to get the folder name
            parsed_url = urlparse(url)
            folder_name = os.path.basename(parsed_url.path)

            # Create the folder if it doesn't exist
            if add_base:
                folder_path = os.path.join(save_path, folder_name)
            else:
                folder_path = save_path
            os.makedirs(folder_path, exist_ok=True)

            # Get the contents of the folder
            response = requests.get(url)
            if response.status_code == 200:
                for item in response.json():
                    if item["type"] == "file" and item["name"].endswith(
                        ".csv"
                    ):
                        print(f"Downloading {item['name']: <79s}", end="\r")
                        file_url = item["download_url"]
                        file_name = os.path.basename(p=file_url)
                        file_path = os.path.join(folder_path, file_name)
                        with open(file_path, "wb") as file:
                            file.write(requests.get(file_url).content)
                    elif item["type"] == "dir":
                        download_csv_from_git(item["url"], folder_path)

        # Example usage
        download_csv_from_git(url, file_path, add_base=False)

    # Recursively go through directories in file_path
    data_dict = {}
    for root, _, files in os.walk(file_path):
        # Create a dictionary to store the data frames
        relative_path = os.path.relpath(root, file_path)
        if relative_path != ".":
            data_dict[relative_path] = []
            for file in files:
                if file.endswith(".csv"):
                    # Get the relative path of the file
                    # Create the corresponding directory structure in the dictionary
                    df = pd.read_csv(
                        os.path.join(root, file), index_col=0, sep=";"
                    )
                    # Store the data frame in the dictionary
                    data_dict[relative_path].append(df)
    # Return the data dictionary
    return data_dict


def load_usp(
    file_path: str = "data/usp-stream-data",
) -> dict[str, pd.DataFrame]:
    from scipy.io.arff import loadarff

    url = (
        "http://sites.labic.icmc.usp.br/vsouza/repository/usp-stream-data.zip"
    )

    if not os.path.exists(file_path):
        response = requests.get(url)
        if response.status_code == 200:
            raise NotImplementedError(
                f"Please, download the data from the following URL: {url}.\n"
                "Feel free to contribute by implementing the download process."
            )

    # Recursively go through directories in file_path
    data_dict: dict[str, pd.DataFrame] = {}
    for root, _, files in os.walk(file_path):
        # Create a dictionary to store the data frames
        if root != ".":
            for file in files:
                if file.endswith(".arff"):
                    # Get the relative path of the file
                    # Create the corresponding directory structure in the dictionary
                    raw_data, meta = loadarff(os.path.join(root, file))
                    df = pd.DataFrame(raw_data, columns=meta.names())
                    # Store the data frame in the dictionary
                    data_dict[file.split(".")[0]] = df
    return data_dict
