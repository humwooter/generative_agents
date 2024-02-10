import requests

# def send_request(input_text):
#     url = "http://<PC_IP_ADDRESS>:8000/predict/"
#     data = {"input_text": input_text}
#     response = requests.post(url, json=data)
#     return response.json()



def call_llm_model(server_url, input_text, **hyperparameters):
    # use server_url for the API request
    payload = {
        "input_text": input_text,
        **hyperparameters
    }
    response = requests.post(server_url, json=payload)  # Use server_url here
    if response.status_code == 200:
        data = response.json()
        return data.get("generated_text", "No generated text found")
    else:
        return f"Error: Could not generate prediction, status code {response.status_code}"

if __name__ == "__main__":
    result = send_request("Hello, world!")
    print(result)