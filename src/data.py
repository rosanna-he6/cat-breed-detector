import os
import zipfile

KAGGLE_OWNER = "imbikramsaha"
KAGGLE_DATASET = "cat-breeds"

DOWNLOAD_DIR = "data"
EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, "processed")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

print("Downloading dataset")
os.system(f'kaggle datasets download -d {KAGGLE_OWNER}/{KAGGLE_DATASET} -p {DOWNLOAD_DIR} --force')

#find zip file
zip_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.zip')]
if not zip_files:
    raise FileNotFoundError("No zip file found in download directory")
zip_path = os.path.join(DOWNLOAD_DIR, zip_files[0])

print("extracting dataset")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Dataset in: ", EXTRACT_DIR)