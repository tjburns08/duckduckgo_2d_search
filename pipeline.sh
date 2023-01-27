#!/bin/sh

# This script is used to run the pipeline for the project.
# When you search for a multi-word query, you should use quotes.
python search_and_bert.py $1
open http://127.0.0.1:8050/
python app.py
