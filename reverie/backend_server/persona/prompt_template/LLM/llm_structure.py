"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import openai
import time 
import requests
import LocalLLM


from utils import *

openai.api_key = openai_api_key

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt): 
  temp_sleep()

  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
  )
  return completion["choices"][0]["message"]["content"]


# ============================================================================
# #####################[SECTION 1: LLM STRUCTURE] ######################
# ============================================================================


# usage: 
# prompt = "Hello, world!"
# model_name = "gpt-4"  # Or any other model you have set up
# server_url = "http://localhost:8000"  # URL of your local server
# response = LLM_request(prompt, model_name, server_url)
def LLM_request(prompt, model_name="gpt-4", server_url="http://localhost:8000"): 
    """
    Sends a request to a locally run LLM via a custom wrapper.
    
    ARGS:
        prompt: a str prompt
        model_name: the name of the model to use (e.g., "gpt-4")
        server_url: the URL of the local server handling the requests
    
    RETURNS: 
        a str of the LLM's response.
    """
    # Initialize the custom LLM with the model name and server URL
    local_llm = LocalLLM(model_name=model_name, server_url=server_url)
    
    try: 
        # Use the custom LLM to generate a response
        response = local_llm.invoke(prompt)
        return response
    except Exception as e: 
        print(f"LLM Request ERROR: {e}")
        return "LLM Request ERROR"

def LLM_safe_generate_response(prompt, example_output, special_instruction, repeat=3, fail_safe_response="error", func_validate=None, func_clean_up=None, verbose=False, model_name="gpt-4", server_url="http://localhost:8000"):
  """
  **Safely generates a response from a local LLM** with retries and validation, using LangChain.

  **ARGS**:
  - prompt: The input string for the LLM.
  - example_output: An example of the expected output format.
  - special_instruction: Additional instructions for the LLM.
  - repeat: Number of retry attempts, defaulting to 3.
  - fail_safe_response: Default error response.
  - func_validate: Optional validation function for the response.
  - func_clean_up: Optional cleanup function for the response.
  - verbose: Enables detailed logging if True.
  - model_name: Specifies the LLM model, defaulting to "gpt-4".
  - server_url: The local server URL.

  **RETURNS**:
  - The processed LLM response or a fail-safe response.
  """
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      local_llm = LocalLLM(model_name=model_name, server_url=server_url)
      curr_gpt_response = local_llm.invoke(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = json.loads(curr_gpt_response[:end_index])["output"]

      if func_validate and func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt) if func_clean_up else curr_gpt_response

      if verbose:
        print(f"**Repeat count**: {i}\n{curr_gpt_response}\n~~~~")

    except Exception as e:
      if verbose:
        print(f"**Attempt {i+1} failed with error**: {e}")
      pass

  if verbose:
    print("**FAIL SAFE TRIGGERED**")
  return fail_safe_response


def generate_prompt(curr_input, prompt_lib_file): 
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
                           model_name="gpt-4",
                           server_url="http://localhost:8000",
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
    if verbose: 
        print(prompt)

    for i in range(repeat):
        curr_llm_response = LLM_request(prompt, model_name, server_url)
        if func_validate and func_validate(curr_llm_response, prompt=prompt):
            return func_clean_up(curr_llm_response, prompt=prompt) if func_clean_up else curr_llm_response
        if verbose:
            print("---- repeat count: ", i, curr_llm_response)
            print(curr_llm_response)
            print("~~~~")
    return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
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

  print (output)




















