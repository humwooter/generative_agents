from typing import Any, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
import requests

class LocalLLM(LLM):
    def __init__(self, model_name: str, server_url: str):
        self.model_name = model_name
        self.server_url = server_url

    @property
    def _llm_type(self) -> str:
        return "Local"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            **kwargs
        }
        response = requests.post(f"{self.server_url}/generate", json=payload)
        if response.status_code == 200:
            return response.json().get("content", "No content returned")
        else:
            raise Exception(f"Server error: {response.status_code}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "server_url": self.server_url}