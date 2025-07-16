import pandas as pd

# Load the data from a CSV file
file_path = '/home/intra.cea.fr/ao280403/Bureau/ML Model/Data/Data B Ph V0.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)

# Compute statistics
description = df.describe()

# Print the statistics to the terminal
print(description)