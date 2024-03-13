import sys
sys.path.append("..")

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import llama_chat
import asyncio
import httpx

app = FastAPI()

# Assuming you might still want to allow CORS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/run_ggml_model")
async def run_ggml_model_endpoint(model_path: str = Query(...), userInput: str = Query(...), chat_uuid: str = Query(...), prompt_intro: str = Query(...)):
    """
    Endpoint to run the GGML model with given parameters.
    
    Args:
    - model_path: The filesystem path to the model.
    - userInput: User input text for the model to respond to.
    - chat_uuid: Unique identifier for the chat session.
    - prompt_intro: Introduction text for the model prompt.
    
    Returns:
    A JSON response containing the model's result.
    """
    result = llama_chat.run_ggml_model(model_path, userInput, chat_uuid, prompt_intro)
    return {"result": result}

@app.get("/kill_chat")
async def kill_chat_endpoint(chat_uuid: str = Query(...)):
    """
    Endpoint to terminate a chat session.
    
    Args:
    - chat_uuid: Unique identifier for the chat session to be terminated.
    
    Returns:
    A JSON response indicating the chat termination status.
    """
    llama_chat.kill_chat(chat_uuid)
    return {"message": "Chat terminated successfully."}
