from duckduckgo_search import ddg
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as cluster
from keybert import KeyBERT
import nltk
import sys
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

# Search for anything
clust_method = 'kmeans'

# Set keywords as command line argument
print("searching for: " + ' '.join(sys.argv[1:]) + "...")
keywords = ' '.join(sys.argv[1:])

to_display = 'body' # Sometimes this is title
md = ddg(keywords, region='wt-wt', safesearch='Moderate', time='y', max_results=500)
md = pd.DataFrame(md)

# Load the model
print("running sentence embeddings...")
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)
sentence_embeddings = model.encode(md['body'].tolist(), show_progress_bar = True)
sentence_embeddings = pd.DataFrame(sentence_embeddings)

# Reduce dimensionality
print("reducing dimensionality...")
reducer = umap.UMAP(metric = 'cosine')
dimr = reducer.fit_transform(sentence_embeddings)
dimr = pd.DataFrame(dimr, columns = ['umap1', 'umap2'])

# Cluster
print("clustering...")
labels = cluster.KMeans(n_clusters=10).fit_predict(dimr[['umap1', 'umap2']])
dimr['cluster'] = labels
dat = pd.concat([dimr.reset_index(), md.reset_index()], axis = 1)

# Add keywords to the clusters
# Create WordNetLemmatizer object
print('extracting keywords per cluster...')
wnl = WordNetLemmatizer()
kw_model = KeyBERT()

keywords_df = []
for i in np.unique(dat['cluster']):
    curr = dat[dat['cluster'] == i]
    text =  ' '.join(curr['title'])
    
    # Lemmatization
    text = nltk.word_tokenize(text)
    text = [wnl.lemmatize(i) for i in text]
    text = ' '.join(text)
    
    # Keyword extraction
    TR_keywords = kw_model.extract_keywords(text)
    keywords_df.append(TR_keywords[0:10])
    
keywords_df = pd.DataFrame(keywords_df)
keywords_df['cluster'] = np.unique(dimr['cluster'])
keywords_df.columns = ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'cluster']

# Merge the keywords with the clusters
dat = dat.merge(keywords_df) # This messes up the index, so we need to reset it
dat = dat.reset_index(drop = True)

labels_cat = dat['cluster'].astype("category")
columns = ['title', 'href', 'body'] + ['keyword' + str(i) for i in range(1, 6)] 

# Ouput as csv)
dat.to_csv('ddg_search_results.csv')
print('data successfully saved to ddg_search_results.csv')




