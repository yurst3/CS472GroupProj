import pandas as pd
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import os
import requests

# Fraction of entries to use from the catalogue
catalogue_fraction = 1

# Only download if the this is the type of artwork
# Set to 'None' to download everything
restrict_form = "painting"

# Skips over files if they are already downloaded
skip_downloaded = True

# Resize each downloaded image to these dimensions
resize_dimensions = (16, 16)

# Name of the directory to download everything to
directory = f'images_{resize_dimensions[0]}x{resize_dimensions[1]}'

# Check to make sure the directory actually exists
if not os.path.exists(directory):
    os.mkdir(directory)

# Read in catalog
catalog = pd.read_csv("catalog.csv", encoding='latin')

# Filter out data entries that don't match the form
if restrict_form is not None:
    catalog = catalog[catalog.FORM == restrict_form]

# Calculate the total number of entries to use and trim catalog to that length
total = int(len(catalog['URL']) * catalogue_fraction)
catalog = catalog[:total]

# Use progress bar from tqdm library
with tqdm(total=total) as pbar:
    for index, row in catalog.iterrows():

        # Get URL and modify so it requests the jpg image instead of an html page
        # EX: https://www.wga.hu/html/a/aachen/adonis.html --> https://www.wga.hu/art/a/aachen/adonis.jpg
        url = row['URL'].replace("/html", "/art").replace(".html", ".jpg")
        author = row['AUTHOR']
        title = row['TITLE']

        # Create file path using index
        jpg_file_path = directory + f"/{index}.jpg"

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

        # Open file and write contents of response to it
        try:
            with open(jpg_file_path, "wb") as image_jpg:
                image_jpg.write(response.content)
                im = Image.open(jpg_file_path)

                # Resize image to dimensions specified above
                resized_im = im.resize(resize_dimensions)
                resized_im.save(jpg_file_path)

        # If PIL can't open the image for some reason, remove it from the directory
        except UnidentifiedImageError:
            pbar.write(f'Error opening {jpg_file_path}, removing from directory')
            os.remove(jpg_file_path)

        pbar.update(1)