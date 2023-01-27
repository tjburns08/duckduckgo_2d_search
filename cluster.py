'''
Description: Takes in tweet data that must include xy dimension reduction coordinates, and performs clustering on the map. 
For each cluster, keywords are extracted to be used for downstream annotation. 
Author: Tyler J Burns
Date: November 2, 2022
'''

from lib2to3.pgen2.pgen import DFAState
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
import nltk
import sklearn.cluster as cluster

# TODO solve the hdbscan bug. 
# Relevant link: https://stackoverflow.com/questions/73830225/init-got-an-unexpected-keyword-argument-cachedir-when-importing-top2vec
clust_method = 'kmeans'

# Must contain umap coordinates
df = pd.read_feather('tmp.feather')

# Dev
# df = df.head(10000)

# We are using dbscan rather than hdbscan at the moment due to a bug from one of the libraries being the wrong version. 
if clust_method == "dbscan":
    
    '''
    Hdbscan is busted and has not been updated
    import hdbscan
    labels = hdbscan.HDBSCAN(
        min_samples=5,
        min_cluster_size=20,
    ).fit_predict(dimr[['umap1', 'umap2']])
    '''
    labels = cluster.DBSCAN(eps=3, min_samples=2).fit_predict(df[['umap1', 'umap2']])
else:
    labels = cluster.KMeans(n_clusters=50).fit_predict(df[['umap1', 'umap2']])
    
    # We want meanshift but it runs too slow for our larger datasets. 
    # TODO find a faster version of meanshift. Use cython or numba?
    # labels = cluster.MeanShift(min_bin_freq = 100).fit_predict(dimr[['umap1', 'umap2']])

df['cluster'] = labels
print(df.cluster[1:10])

# For initial use of the script
#nltk.download('punkt')
#nltk.download('omw-1.4')
#nltk.download('wordnet')


# Here is the keyword extraction. We're using KeyBERT simply because it's consistent with sBERT, which we use to do the embeddings
# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()
kw_model = KeyBERT()

keywords_df = []
for i in np.unique(df['cluster']):
    curr = df[df['cluster'] == i] 
    text =  ' '.join(curr['Tweet']) 
    
    # Lemmatization
    text = nltk.word_tokenize(text)
    text = [wnl.lemmatize(i) for i in text]
    text = ' '.join(text)
    
    # Keyword extraction
    TR_keywords = kw_model.extract_keywords(text)
    keywords_df.append(TR_keywords[0:10])
    
keywords_df = pd.DataFrame(keywords_df)
keywords_df['cluster'] = np.unique(df['cluster'])
keywords_df.columns = ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'cluster']
print(keywords_df)

# Combine with original dataframe
df = df.merge(keywords_df, on = 'cluster', how = 'left') # This messes up the index

print(df.cluster[1:10])
print(df.keyword1[1:10])

df.to_csv('tmp.csv', index = False)