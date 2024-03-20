"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import openai
import time 

from utils import *
from persona.prompt_template.print_prompt import *


# client = anthropic.Anthropic(
#     api_key= anthropic_api_key
# )

openai.api_key = openai_api_key

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt):
  log_and_track_function_calls( "ChatGPT_single_request") # REMOVE LATER              
  temp_sleep()

  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
  )
  return completion["choices"][0]["message"]["content"]


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  log_and_track_function_calls( "GPT4_request") # REMOVE LATER            
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try: 
    completion = openai.ChatCompletion.create(
    model="gpt-4", 
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
  
  except: 
    #print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt):
  simple_write_to_file("ChatGPT_request-prompts.txt", prompt) # REMOVE LATER             
  log_and_track_function_calls( "ChatGPT_request") # REMOVE LATER            
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try: 
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
  
  except: 
    #print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  log_and_track_function_calls( "GPT4_safe_generate_response") # REMOVE LATER            
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    #print ("CHAT GPT PROMPT")
    #print (prompt)
    write_to_file_in_console_logs(filename="prompts.txt", text=prompt)


  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        #print ("---- repeat count: \n", i, curr_gpt_response)
        write_to_file_in_console_logs(filename="curr-gpt-response.txt", text=curr_gpt_response)
        #print (curr_gpt_response)
        #print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  log_and_track_function_calls( "ChatGPT_safe_generate_response") # REMOVE LATER            
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    for i in range(repeat): 
      try: 
        curr_gpt_response = ChatGPT_request(prompt).strip()
        end_index = curr_gpt_response.rfind('}') + 1
        curr_gpt_response = curr_gpt_response[:end_index]
        curr_gpt_response = json.loads(curr_gpt_response)["output"]

        
        if func_validate(curr_gpt_response, prompt=prompt): 
          return func_clean_up(curr_gpt_response, prompt=prompt)
        
        if verbose: 
          #print ("---- repeat count: \n", i, curr_gpt_response)
          #print (curr_gpt_response)
          # write_to_file_in_console_logs(filename="curr-gpt-response.txt", text=curr_gpt_response)
          return curr_gpt_response #####
          #print ("~~~~")

      except: 
        pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  log_and_track_function_calls( "ChatGPT_safe_generate_response_OLD") # REMOVE LATER            
  if verbose: 
    #print ("CHAT GPT PROMPT")
    #print (prompt)
    write_to_file_in_console_logs(filename="prompts.txt", text=prompt)


  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        #print (f"---- repeat count: {i}")
        #print (curr_gpt_response)
        write_to_file_in_console_logs(filename="curr-gpt-response.txt", text=curr_gpt_response)

        #print ("~~~~")

    except: 
      pass
  #print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================



def Claude_request(prompt, parameters):
  log_and_track_function_calls("Claude_request")
  save_gpt_prompt_to_file("Claude_request-prompts.txt", parameters, prompt)
  

  # Map the gpt_parameter dictionary to Claude's parameters
  model =  "claude-3-opus-20240229"  # Default to a model if not specified
  max_tokens = parameters.get("max_tokens", 1000)
  temperature = parameters.get("temperature", 0)  # Claude's default temperature if not specified
  # Add additional mappings as necessary

  try:
      # Creating the message with Claude
      message = client.messages.create(
          model=model,
          max_tokens=max_tokens,
          temperature=temperature,
          system=prompt,  # Directly use the 'prompt' as the 'system' input for Claude
          messages=[
              {
                  "role": "user",
                  "content": [
                      {
                          "type": "text",
                          "text": prompt
                      }
                  ]
              }
          ]
      )
      # Assuming 'message.content' holds the response text
      return message.content
  except Exception as e:
      print(f"An error occurred: {e}")
      return "Request failed"


def LLM_request(prompt, llm_parameters, url="http://169.231.53.160:1234/v1/completions", model_name="TheBloke_Llama-2-7B-fp16"):
  import requests
  log_and_track_function_calls( "LLM_request")
  save_gpt_prompt_to_file("LLM_request-prompts.txt", llm_parameters, prompt)
  
  # Setting headers for the request
  headers = {
      "Content-Type": "application/json"
  }
  
  # Preparing the data payload for the request, including all LLM parameters and optional model name
  data = {
      "prompt": prompt,
      "model": model_name,  # Use the model name if specified
      "max_tokens": llm_parameters.get("max_tokens", 200),
      "temperature": llm_parameters.get("temperature", 1.0),
      "top_p": llm_parameters.get("top_p", 1.0),
      "frequency_penalty": llm_parameters.get("frequency_penalty", 0),
      "presence_penalty": llm_parameters.get("presence_penalty", 0),
      "stream": llm_parameters.get("stream", False),
      "stop": llm_parameters.get("stop", None)
  }
  
  try:
      # Sending the request to the specified LLM server URL
      response = requests.post(url, headers=headers, json=data, verify=False)
      # Parsing the response
      response_data = response.json()
      return response_data['choices'][0]['text']
  except Exception as e:
      print(f"An error occurred: {e}")
      return "Request failed"

def get_wake_up_hour(json_response):
  import re
  json_string = json_response["response"]
  try:
    data = json.loads(json_string)
    for key, value in data.items():
        # Check if key contains 'wake'
        if "wake" in key.lower():
            return value
        # Check if value is a string that contains 'am'
        if isinstance(value, str) and "am" in value.lower():
            return value
        # Check if value matches the format '0H:MM'
        if isinstance(value, str) and re.match(r'\d{1}:\d{2}', value):
            return value
    return ""
  except json.JSONDecodeError:
      return "Invalid JSON input"
  return ""

def GPT_request_with_local_fallback(prompt, gpt_parameter):
  log_and_track_function_calls("GPT_request_with_local_fallback")
  save_gpt_prompt_to_file("GPT_request_with_local_fallback-prompts.txt", gpt_parameter, prompt)
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. If the prompt length is less than 300 words,
  a local request is made instead.
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response or local model's response. 
  """
  print("PROMPT FROM GPT_REQUEST: ", prompt)
  temp_sleep()
  if gpt_parameter["max_tokens"] == 5:
    print("ENTEERED THIS LOOP WHERE IT SAYS THERES 5 TOKENS")

    response_json = send_request_to_localhost(prompt, "gemma:2b")
    response_str = str(get_wake_up_hour(response_json)).replace(" ", "")
    json_string = response_json["response"]

    if len(response_str) == 0:
      try: 
        final_response = GPT_request_standard(prompt, gpt_param)
        save_gpt_output_to_file("GPT_request-complete-output.txt", gpt_parameter, prompt, final_response)
        return final_response

      except: 
        return "TOKEN LIMIT EXCEEDED"
    else:
      formatted_output = f"{json_string}\n{response_str}"

      save_gpt_output_to_file("Gemma_request-complete-output.txt", gpt_parameter, prompt, formatted_output)
      log_and_track_function_calls("Gemma_request_with_local_fallback")
      return response_str
  else:
    print("ENTERED THE OTHER LOOP WHERE TOKENS IS MORE THAN 5")
    try:
      final_response = GPT_request_standard(prompt, gpt_parameter)
      save_gpt_output_to_file("GPT_request-complete-output.txt", gpt_parameter, prompt, final_response)
      print("RESPOSNES FROM WITHIN LOOP 2: ", final_response)

      return final_response
    except: 
      return "TOKEN LIMIT EXCEEDED"


def send_request_to_localhost(prompt, model_name):
    import requests
    import json
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print("RESPONSE: ", response.json())  # This line is changed to print the JSON content of the response
    return response.json()  # Assuming the response JSON structure contains the generated text under a "text" key


def GPT_request(prompt, gpt_parameter):
  return GPT_request_with_local_fallback(prompt, gpt_parameter) ######

  
def GPT_request_standard(prompt, gpt_parameter):
  log_and_track_function_calls( "GPT_request")
  save_gpt_prompt_to_file("GPT_request-prompts.txt", gpt_parameter, prompt)        
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try: 
    response = openai.Completion.create(
                model=gpt_parameter["engine"],
                prompt=prompt,
                temperature=gpt_parameter["temperature"],
                max_tokens=gpt_parameter["max_tokens"],
                top_p=gpt_parameter["top_p"],
                frequency_penalty=gpt_parameter["frequency_penalty"],
                presence_penalty=gpt_parameter["presence_penalty"],
                stream=gpt_parameter["stream"],
                stop=gpt_parameter["stop"],)
    save_gpt_output_to_file("GPT_request-complete-output.txt", gpt_parameter, prompt, response.choices[0].text)
    return response.choices[0].text
  except: 
    return "TOKEN LIMIT EXCEEDED"

def generate_prompt(curr_input, prompt_lib_file):
  log_and_track_function_calls( "generate_prompt") # REMOVE LATER             

  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
  print("PROMPT FROM SAFE GENERATE RESPONSE: ", prompt)

  if verbose: 
    log_and_track_function_calls( "safe_generate_response") # REMOVE LATER             
    write_to_file_in_console_logs(filename="prompts.txt", text=prompt)


  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    print("RESPONSE FROM SAFE GENERATE RESPONSE: ", curr_gpt_response)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      #print ("---- repeat count: ", i, curr_gpt_response)
      #print (curr_gpt_response)
      write_to_file_in_console_logs(filename="curr-gpt-response.txt", text=curr_gpt_response)

      #print ("~~~~")
  return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
  log_and_track_function_calls( "get_embedding") # REMOVE LATER             
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return openai.Embedding.create(
          input=[text], model=model)['data'][0]['embedding']


if __name__ == '__main__':
  gpt_parameter = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  