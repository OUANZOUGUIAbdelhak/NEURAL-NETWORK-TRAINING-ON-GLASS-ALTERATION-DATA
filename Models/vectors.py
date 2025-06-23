import gensim.downloader
import matplotlib.pyplot as plt

# Load the GloVe model
model = gensim.downloader.load("glove-wiki-gigaword-50")

# Get the vector for the word "house"
house_vector = model["house"]

# Plot the vector components
plt.figure(figsize=(10, 6))
plt.bar(range(len(house_vector)), house_vector)
plt.title("Word Vector for 'house'")
plt.xlabel("Vector Component Index")
plt.ylabel("Component Value")

# Save the plot to a file
plt.savefig("house_vector_plot.png")
