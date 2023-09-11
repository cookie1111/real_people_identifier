import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from sklearn.cluster import KMeans
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD



vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)


# Define the path to the main folder and the captions subfolder
main_folder = '/media/sebastjan/663C012C3C00F8B7/Users/Sebastjan/Downloads/InstaCities1M/InstaCities1M'
caption_folder = os.path.join(main_folder, 'captions')
subfolders = ['train', 'test', 'val']
cities = ['chicago', 'london', 'losangeles', 'melbourne', 'miami', 'newyork', 'sanfrancisco', 'singapore', 'sydney', 'toronto']
"""
# Loop through the subfolders and cities
for subfolder in subfolders:
    for city in cities:
        df = pd.DataFrame(columns=["id", "caption"])
        caption_path = os.path.join(caption_folder, subfolder, city)

        # Get the list of files in the directory
        files = sorted(os.listdir(caption_path))

        # Loop through the images in this city's folder with a loading bar
        for filename in tqdm(files, desc=f"Processing {city}"):
            # Only process .txt files
            if filename.endswith('.txt'):
                # Read the caption from the file
                with open(os.path.join(caption_path, filename), 'r') as f:
                    caption = f.read()

                # Add the image id and the corresponding description to the DataFrame
                df = df.append({"id": filename[:-4], "caption":caption}, ignore_index=True)

        # Save the DataFrame as a CSV file with the name of <city>_caption.csv
        df.to_csv(f"{subfolder}_{city}_caption.csv", index=False)"""


for city in cities:
    # Create a list to store the DataFrames
    dfs = []

    # Loop through the CSV files
    for csv_file in [f"{folder}_{city}_caption.csv" for folder in subfolders]:
        # Load the data from the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate the DataFrames into one DataFrame
    all_data = pd.concat(dfs, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    #all_data.to_csv('all_chicago_caption.csv', index=False)

    # Convert the captions to a matrix of TF-IDF features
    X = vectorizer.fit_transform(all_data['caption'])

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=15)  # choose the appropriate number of clusters
    kmeans.fit(X)

    # Add the cluster labels to your DataFrame
    all_data['cluster'] = kmeans.labels_

    # Save the DataFrame with the cluster labels
    all_data.to_csv(f'output_with_clusters_{city}.csv', index=False)

    # Reduce the dimensionality of the data using TruncatedSVD
    svd = TruncatedSVD(n_components=50)
    X_reduced = svd.fit_transform(X)

    # Sample a subset of your data
    sample_indices = np.random.choice(X_reduced.shape[0], size=10000, replace=False)
    X_sample = X_reduced[sample_indices]
    cluster_sample = all_data['cluster'].iloc[sample_indices]

    # Project the data into 2D space using t-SNE
    tsne = TSNE(n_components=2)
    X_2d = tsne.fit_transform(X_sample)

    # Create a scatter plot of the projected data
    plt.figure(figsize=(10, 10))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_sample)
    plt.title(f'Clusters for {city}')
    plt.show()

    # Print some examples from each cluster
    for cluster in range(5):
        print(f'Cluster {cluster}:')
        print(all_data[all_data['cluster'] == cluster]['caption'].head())