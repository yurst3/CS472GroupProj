import pandas as pd
import random

file = pd.read_csv("catalog.csv", encoding='latin')

# Print random author name
print(file["AUTHOR"][random.randint(0, len(file["AUTHOR"]))])