#!/bin/sh

# This script is used to run the pipeline for the project.
# When you search for a multi-word query, you should use quotes.
python search_and_bert.py $1
python app.py
