#!/bin/bash

# Create a virtual environment
python3 -m venv llm_server_env
source llm_server_env/bin/activate

# Install FastAPI and Uvicorn
pip3 install fastapi uvicorn

# Additional model dependencies
# pip install <your-model-dependencies>

# Run the server (assuming your FastAPI file is named app.py and located under server/)
uvicorn server.app:app --host 0.0.0.0 --port 8000