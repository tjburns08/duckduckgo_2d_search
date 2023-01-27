'''
Description: Takes in scraped tweets metadata and products per-tweet high dimensional sentence embeddings.
Author: Tyler J Burns
Date: November 2, 2022
'''

from logging import error
import pandas as pd
from sentence_transformers import SentenceTransformer
import sys
import subprocess
from datetime import datetime
import glob
from dateutil.parser import parse

# TODO fix the indexing bug
# TODO fix date moves to another location bug
# We jump into an adjacent project
# df = pd.read_csv('../twitter_archive_scrape/output/' + user + '_tweets_scraped.csv')

def transform_sentence(tweet_df):
    '''
    Description: Takes a tweet data frame as input and produces a data frame of sentence embeddings in proper order.
    Args:
        tweet_df: Data frame that contains at least a column called Tweet, which is the sentence to be transformed.
    Returns: A data frame of sentence embeddings
    '''
    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    se = model.encode(tweet_df['Tweet'].tolist(), show_progress_bar = True)
    se = pd.DataFrame(se)
    se.columns = se.columns.astype(str)
    return se

# This is the list of twitter handles
users = pd.read_csv('../twitter_archive_scrape/users.csv') # From adjacent project
users = users['user'].to_list()

# users = ['cnnbrk'] # Dev
curr_files = glob.glob('output/*')
print(curr_files)
now = datetime.now()
now = now.strftime('%Y-%m-%d')

count = 0
for i in users:
    # The block we're gonna loop
    print(i)
    file = i + '_tweets_scraped.csv'

    # Calls an R script that fixes a strange row bug that we get from using Pandas to scrape the tweets
    # All this script does is read in the DataFrame as an R Data Frame object and output it again. 
    # This seems to fix the bug on its own. 
    input_str = 'Rscript fix_row_bug.R' + ' ' + file
    subprocess.call(input_str, shell=True)
    
    # This csv file is output from the fix_row_bug.R file above
    df = pd.read_csv('tmp.csv')
    if df.shape[0] == 0:
        continue
    # df = df.head(100) for dev 
    print(df)
    print(df['User'])

    # If we've already transformed the sentences, then only transform the new tweets
    if any([i + '_' in f for f in curr_files]):
        print('User in curr_files')
        se_orig = pd.read_feather('output/' + i + '_sentence_embeddings.feather')
        print(se_orig)

        # Bug from original scrape tweets process
        if se_orig.shape[0] == 0:
            continue

        # Get most recent date of the current sentence embeddings file
        first = parse(se_orig['Date'][0])
        # first = first.strftime('%Y-%m-%d')
        print('first')
        print(first)

        # TODO set this for the scrape tweets script, to get rid of duplicates
        is_new_date = [parse(i) > first for i in df['Date']]
        df = df[is_new_date]
        print(df)
        if df.shape[0] == 0:
            continue

    # This removes the hyperlink that is often added to the end of a tweet especially for news media handles
    df['Tweet'] = [i.split('http')[0] for i in df['Tweet']]
    print('debug check')
    print(df)
    se = transform_sentence(df)
    print(se)
    
    # Sanity checks
    if se.shape[0] != df.shape[0]:
        error('Number of rows of tweet dataframe and embedded sentences are different. Please fix this and run again')
    
    # Warning: possible bug here if weird 'unknown index' columns pop up
    # Solution: explicitly add the columns you care about from the original data frame
    se = se.reset_index(drop = True).join(df.reset_index(drop = True))
   
    print(se['Date'])

    # Merge with se_orig
    if any([i + '_' in f for f in curr_files]):
        # TODO change this, as append is depricated
        se = se.append(se_orig, ignore_index = True)
        print(se.shape[0])

    # This is the output that gets passed to the next step in the pipeline
    se.to_feather('output/' + i + '_sentence_embeddings.feather')
    
    # This is redundant, but it allows the user to check in Excel whether the embeddings look right
    if count < 5:
        se.to_csv('output/' + i + '_sentence_embeddings.csv')
    count += 1
       






