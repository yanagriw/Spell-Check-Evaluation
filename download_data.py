import os
import requests


def download_file(url, save_path):
    """
    Downloads a file from a given URL and saves it to the specified path if it doesn't already exist.

    :param url: The URL to download the file from.
    :param save_path: The local path where the file will be saved.
    """
    # Check if the file already exists
    if not os.path.exists(save_path):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Write the file content
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded successfully and saved as {save_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download the file. Error: {e}")
    else:
        print(f"File already exists at {save_path}")

def main():
    # URLs to download
    url1 = "https://www.dcs.bbk.ac.uk/~roger/missp.dat"
    url2 = "https://www.dcs.bbk.ac.uk/~roger/aspell.dat"
    url3 = "https://www.dcs.bbk.ac.uk/~roger/holbrook-tagged.dat"

    # File save paths
    save_path1 = "data/birkbeck.dat"
    save_path2 = "data/aspell.dat"
    save_path3 = "data/holbrook.dat"

    # Download files only if they don't exist
    download_file(url1, save_path1)
    download_file(url2, save_path2)
    download_file(url3, save_path3)

if __name__ == "__main__":
    main()