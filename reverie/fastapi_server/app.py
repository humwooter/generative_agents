from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RequestModel(BaseModel):
    input_text: str
    server_url: str  # add this line
    temperature: float = 0.7  
    max_tokens: int = 100
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@app.post("/predict/")
async def make_prediction(request: RequestModel):
    # extract hyperparameters from the request
    hyperparameters = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty
    }
    
    # pass server_url along with the input text and hyperparameters
    prediction = call_llm_model(request.server_url, request.input_text, **hyperparameters)
    
    response = {"result": prediction}
    return response


