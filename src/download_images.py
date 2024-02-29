import requests
from tqdm import tqdm
import zipfile


def download_file(url: str, filename: str) -> None:
    """
    Downloads a file from a given URL and saves it to a local path, displaying the download progress.

    :param url: str, the URL to download the file from.
    :param filename: str, the local path where the file will be saved.
    """
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


coco_train_images_url = "http://images.cocodataset.org/zips/val2017.zip"
download_file(coco_train_images_url, "coco_val2017.zip")

with zipfile.ZipFile("coco_val2017.zip", 'r') as zip_ref:
    zip_ref.extractall("test_images")
