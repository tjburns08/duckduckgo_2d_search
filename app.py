'''
Description: his creates the web app. It runs Dash on a local server. 
The user runs this script after the dimension reduction and clustering has been run on the orignal tweet metadata file. 
Author: Tyler J Burns
Date: November 2, 2022
'''

from dash import Dash, dcc, html, dash_table, Input, Output, callback_context, State
import plotly.express as px
import pandas as pd
import json
import re
import numpy as np
from datetime import datetime

# You have to include these two lines if you want it to run in Heroku
app = Dash(__name__)
server = app.server


def search_bar(input, text):
    '''
    Takes a string with or without logical values (AND, OR) as input, runs that on another given string, and returns boolean corresponding to whether the input was in the other string.

    Args:
        input: The search term. 
        text: The text it will be doing the search on.
    Returns:
        boolean value correspoding to whether the input was in the text

    Note: 
        For logic-based search, you can add (all caps) AND or OR. But you can't add both of them. 

    Example:
        search_bar('beer OR wine", "This beer is good") returns True.
        search_bar('beer AND wine", "This beer is good") returns False.
        search_bar('beer AND wine OR cheese', 'This beer is good') returns False, because this function cannot use combos of AND and OR
    '''
    text = text.lower()
    bool_choice = [input.find('AND') != -1, input.find('OR') != -1]
    
    if(sum(bool_choice) == 0):
        result = text.find(input.lower()) != -1
        return(result)
    
    if sum(bool_choice) == 2:
        return(False)
    if bool_choice[0]:
        bool_choice = 'AND'
    elif bool_choice[1]:
        bool_choice = 'OR'

    input = input.split(' ' + bool_choice + ' ')
    input = [i.lower() for i in input]

    if bool_choice  == 'AND':
        result = [all(text.find(i.lower()) != -1 for i in input)]
    elif bool_choice == 'OR':
        result = [any(text.find(i.lower()) != -1 for i in input)]

    return(result[0])

# This comes from search_and_bert.py
df = pd.read_csv('ddg_search_results.csv', lineterminator='\n')

# Get it ready for plotting
df['title'] = df.title.str.wrap(30).apply(lambda x: x.replace('\n', '<br>'))
df['body'] = df.body.str.wrap(30).apply(lambda x: x.replace('\n', '<br>'))

#df['title'] = [re.sub('<br>', ' ', i) for i in df['title']]
#df['body'] = [re.sub('<br>', ' ', i) for i in df['body']]

# We have the url in markdown format, so it only shows up as a hyperlink
df['href'] = ['[Go to result]' + '(' + i + ')' for i in df['href']]

# We get rid of unnecessary columns for downstream processing
df_sub = df[['href', 'title', 'body']]

# This is the layout of the page. We are using various objects available within Dash. 
fig = px.scatter() # The map
app.layout = html.Div([
    html.A(html.P('Click here for instructions'), href="https://tjburns08.github.io/app_instructions.html"),
    dcc.Textarea(
        placeholder='Type keywords separated by AND or OR',
        id = 'user-input',
        value='',
        style={'width': '100%'}
    ),
    html.Button('Submit', id='value-enter'),
    dcc.Graph(
        id='news-map',
        figure=fig
    ),
    html.Plaintext('Info of result you clicked on.'),
    dash_table.DataTable(data = df_sub.to_dict('records'), style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    }, id='news-table', fill_width = False, columns=[{'id': x, 'name': x, 'presentation': 'markdown'} if x == 'href' else {'id': x, 'name': x} for x in df_sub.columns]),

    # Dev
    html.Plaintext('Top results given search term, or all results if no search term.'),
    dash_table.DataTable(data = df_sub.to_dict('records'), style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    }, id='top-table', fill_width = False, columns=[{'id': x, 'name': x, 'presentation': 'markdown'} if x == 'href' else {'id': x, 'name': x} for x in df_sub.columns])

])

# This allows the user to click on a point on the map and get a single entry corresponding to that article
# TODO consider returning the article and its KNN
@app.callback(
    Output('news-table', 'data'),
    Input('news-map', 'clickData'))

def click(clickData):
    if not clickData:
        return
    tweet = clickData['points'][0]['customdata'][0]
    filtered_df = df_sub[df_sub['title'] == tweet]
    filtered_df['title'] = [re.sub('<br>', ' ', i) for i in filtered_df['title']]
    filtered_df['body'] = [re.sub('<br>', ' ', i) for i in filtered_df['body']]
    return filtered_df.to_dict('records')


# This updates the map given the dropdowns and the value entered into the search bar
@app.callback(  
    Output('news-map', 'figure'),
    Input('value-enter', 'n_clicks'),
    State('user-input', 'value'))

def update_plot(n_clicks, input_value):
    user_context = callback_context.triggered[0]['prop_id'].split('.')[0]

    tmp = df

    if(user_context == 'value-enter' or input_value != ''):
        # input_values = input_value.lower().split(',')
        # Keyword logic
        # TODO make this a standalone function
        rel_rows = []

        # TODO add OR to this
        for i in tmp['title']:
            # rel_rows.append(all([v in i.lower() for v in input_values]))
            rel_rows.append(search_bar(input_value, i))
        tmp = tmp[rel_rows]

    tmp['title'] = tmp.title.str.wrap(30).apply(lambda x: x.replace('\n', '<br>'))
    tmp['body'] = tmp.body.str.wrap(30).apply(lambda x: x.replace('\n', '<br>'))  
    fig = px.scatter(tmp, x = 'umap1', y = 'umap2', hover_data = ['title', 'body', 'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5'], color = 'cluster', title = 'Context similarity map of results')
    
    # DarkSlateGrey
    fig.update_traces(marker=dict(line=dict(width=0.1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    return(fig)


@app.callback(
    Output('top-table', 'data'),
    Input('value-enter', 'n_clicks'))

def update_table(n_clicks):
    filtered_df = df # Make local variable. There might be a less ugly way to do this.
    filtered_df['title'] = [re.sub('<br>', ' ', i) for i in filtered_df['title']]
    filtered_df['body'] = [re.sub('<br>', ' ', i) for i in filtered_df['body']]
    return filtered_df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)