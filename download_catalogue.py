import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import requests

# TODO: Exclude stuff that isn't paintings?
catalogue = pd.read_csv("catalog.csv", encoding='latin')

# Fraction of paintings to use from the catalogue
catalogue_fraction = 1/100

# Skips over files if they are already downloaded
skip_downloaded = True

# Resize each downloaded image to these dimensions
resize_dimensions = (128,128)

# Use progress bar from tqdm library
with tqdm(total=int(len(catalogue['URL'])*catalogue_fraction)) as pbar:
    for i in range(int(len(catalogue['URL'])*catalogue_fraction)):

        # Get URL and modify so it requests the jpg image instead of an html page
        # EX: https://www.wga.hu/html/a/aachen/adonis.html --> https://www.wga.hu/art/a/aachen/adonis.jpg
        url = catalogue['URL'][i].replace("/html", "/art").replace(".html", ".jpg")
        author = catalogue['AUTHOR'][i]
        title = catalogue['TITLE'][i]

        # Create file path using the author and title of the artwork
        # Remove extraneous " from the file path
        jpg_file_path = f"images/{author} \'{title}\'.jpg".replace('\"', '')

        # Skip over this file if it has already been downloaded
        if skip_downloaded and os.path.isfile(jpg_file_path):
            pbar.update(1)
            continue

        pbar.set_description(f"Downloading {url}")

        # Send request to url
        response = requests.get(url, allow_redirects=False)

        # Check for proper status code
        if response.status_code != 200:
            print(f"error retrieving {url}")
            pbar.update(1)
            continue

        with open(jpg_file_path, "wb") as image_jpg:
            image_jpg.write(response.content)
            im = Image.open(jpg_file_path)

            # Resize image to dimensions specified above
            resized_im = im.resize(resize_dimensions)
            resized_im.save(jpg_file_path)

        pbar.update(1)