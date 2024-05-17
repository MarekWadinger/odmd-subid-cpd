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


def load_cats(resample_s: None | int = None) -> pd.DataFrame:
    file_path: str = "data/cats/data.csv"
    url = "https://zenodo.org/records/7646897/files/data.parquet"

    if os.path.exists(file_path):
        # Read the data from the file into a numpy array
        df = pd.read_csv(file_path, index_col=0)
    else:

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

        df = download_and_read_parquet_with_progress(url)

        # Check if the directory exists, if not create it
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    df.index = pd.to_datetime(df.index)
    if resample_s is not None:
        df = df.resample(f"{resample_s}s").median().iloc[resample_s:]

    if not os.path.exists(file_path):
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

        download_csv_from_git(url, file_path, add_base=False)

    # Recursively go through directories in file_path
    data_dict: dict[str, list] = {}
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
    from tqdm import tqdm

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

    def convert_dtypes_numeric(df):
        for col in df:
            df[col] = df[col].map(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            # We are only interested in ordinal numeric values
            df[col] = df[col].map(
                lambda x: float(x)
                if isinstance(x, str) and x.isnumeric()
                else x
            )
        return df

    # Recursively go through directories in file_path
    data_dict: dict[str, pd.DataFrame] = {}
    for root, _, files in os.walk(file_path):
        # Create a dictionary to store the data frames
        if root != ".":
            with tqdm(total=len(files), mininterval=1.0) as pbar:
                for file in files:
                    if file.endswith(".arff"):
                        pbar.set_description(f"Loading {file}")
                        # Get the relative path of the file
                        # Create the corresponding directory structure in the dictionary
                        raw_data, meta = loadarff(os.path.join(root, file))
                        df = pd.DataFrame(raw_data, columns=meta.names())
                        # Store the data frame in the dictionary
                        data_dict[file.split(".")[0]] = df
                    pbar.update(1)

    # Rename the class column to "class" for consistency
    data_dict["chess"] = data_dict["chess"].rename(
        columns={"outcome": "class"}
    )
    data_dict["airlines"] = data_dict["airlines"].rename(
        columns={"Delay": "class"}
    )
    data_dict["gassensor"] = data_dict["gassensor"].rename(
        columns={"Class": "class"}
    )
    data_dict["ozone"] = data_dict["ozone"].rename(columns={"Class": "class"})

    # Convert the data types to numeric
    with tqdm(total=len(files), mininterval=1.0) as pbar:
        for k, df in data_dict.items():
            pbar.set_description(f"Processing {k}")
            df = convert_dtypes_numeric(df)
            gt = df["class"].copy(deep=True)
            df = df.select_dtypes(include="number")
            if "class" not in df.columns:
                if any(gt.apply(lambda x: isinstance(x, str))):
                    df["class"] = gt.astype("category").cat.codes
                else:
                    df["class"] = gt
            df.index = pd.to_datetime(df.index, unit="s")
            data_dict[k] = df
            pbar.update(1)
    return data_dict


def load_bess() -> tuple[pd.DataFrame, pd.DataFrame]:
    folder_path: str = "data/kokam"
    X_path: str = f"{folder_path}/kokam_norm.csv"
    y_path: str = f"{folder_path}/kokam_ground_truth.csv"
    base_url: str = "https://raw.githubusercontent.com/MarekWadinger/adaptive-interpretable-ad/main/examples/"

    for path in [X_path, y_path]:
        url = base_url + path
        if not os.path.exists(path):
            os.makedirs(folder_path, exist_ok=True)

            # Read the data from the file into a numpy array
            def download_csv_from_git(url, save_path):
                # Get the contents of the folder
                response = requests.get(url)
                if response.status_code == 200:
                    with open(save_path, "wb") as file:
                        file.write(response.content)

            download_csv_from_git(url, path)

    X = pd.read_csv(X_path, index_col=0)
    X.index = pd.to_datetime(X.index)
    y = pd.read_csv(y_path, index_col=0)
    y.index = pd.to_datetime(y.index)
    return X, y
