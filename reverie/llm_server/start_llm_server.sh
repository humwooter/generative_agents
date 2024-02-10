#!/bin/bash

# navigate to the LLM server directory
cd "$(dirname "$0")"

# create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip3 install -r requirements.txt

# Start the server
uvicorn model_server:app --host 0.0.0.0 --port 8000