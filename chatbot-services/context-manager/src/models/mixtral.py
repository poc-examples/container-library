from pydantic import BaseModel, Field
from langchain.llms import BaseLLM

class MixtralLLM(BaseLLM):

    model_url: str = Field(...)
    headers: dict = Field(default_factory=lambda: {"Content-Type": "application/json"})

    def _generate(self, prompt, **kwargs):
        import requests

        payload = {
            "input_text": prompt,
        }

        response = requests.post(self.model_url, json=payload, headers=self.headers)

        if response.status_code == 200:
            # If the request was successful, extract and return the generated text
            return response.json()['generated_text']
        else:
            # If the request failed, return the status code and error message
            return f"Request failed with status code {response.status_code}: {response.text}"
        
    def _llm_type(self):
        # Return a string that represents the type of this LLM
        return "MixtralLLM"
