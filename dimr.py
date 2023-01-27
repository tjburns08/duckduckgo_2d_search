"""
Description: Takes in transformed sentence file, a high dimensional matrix, and performs dimension reduction on it. 
This reduces the dimensionality to 2. 
Author: Tyler Burns
Date: November 2, 2022
"""

import umap
import pandas as pd

# This file contains the high-dimensional sentence embeddings per tweet
df = pd.read_feather('tmp.feather')
print(df.columns)
cols = [str(i) for i in range(768)] # Number of dimensions in the embedding

st = df[cols]
embedding = umap.UMAP(densmap=True, random_state=42).fit_transform(st)
embedding = pd.DataFrame(embedding, columns = ['umap1', 'umap2'])

# remove the sentence embedding bulk
df = df.drop(cols, axis = 1)

# Adds umap2 and umap2 as columns, and overwrites the orignal feather file
df = pd.concat([df.reset_index(drop=True), embedding], axis=1)
df.to_feather('tmp.feather')