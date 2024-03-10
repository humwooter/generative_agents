"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: print_prompt.py
Description: For printing prompts when the setting for verbose is set to True.
"""
import sys
sys.path.append('../')

import json
import numpy
import datetime
import random
import os
from global_methods import *
from persona.prompt_template.gpt_structure import *
from utils import *

##############################################################################
#                    PERSONA Chapter 1: Prompt Structures                    #
##############################################################################

def print_run_prompts(prompt_template=None, 
                      persona=None, 
                      gpt_param=None, 
                      prompt_input=None,
                      prompt=None, 
                      output=None): 
  print (f"=== {prompt_template}")
  print ("~~~ persona    ---------------------------------------------------")
  print (persona.name, "\n")
  print ("~~~ gpt_param ----------------------------------------------------")
  print (gpt_param, "\n")
  print ("~~~ prompt_input    ----------------------------------------------")
  print (prompt_input, "\n")
  print ("~~~ prompt    ----------------------------------------------------")
  print (prompt, "\n")
  print ("~~~ output    ----------------------------------------------------")
  print (output, "\n") 
  print ("=== END ==========================================================")
  print ("\n\n\n")


def write_run_prompts_to_file(filename: str, 
                              prompt_template=None, 
                              persona=None, 
                              gpt_param=None, 
                              prompt_input=None,
                              prompt=None, 
                              output=None):
    lines = []
    lines.append(f"=== {prompt_template}")
    lines.append("~~~ persona    ---------------------------------------------------")
    lines.append(f"{persona.name}\n")
    lines.append("~~~ gpt_param ----------------------------------------------------")
    lines.append(f"{gpt_param}\n")
    lines.append("~~~ prompt_input    ----------------------------------------------")
    lines.append(f"{prompt_input}\n")
    lines.append("~~~ prompt    ----------------------------------------------------")
    lines.append(f"{prompt}\n")
    lines.append("~~~ output    ----------------------------------------------------")
    lines.append(f"{output}\n")
    lines.append("=== END ==========================================================\n\n\n")

    # Ensure 'console_logs' directory exists and append output to the specified file
    logs_dir = 'console_logs'
    if not os.path.exists(logs_dir):
        print("DOESNT EXIST")
        os.makedirs(logs_dir)

    file_path = os.path.join(logs_dir, filename)
    with open(file_path, 'a') as file:
        print("WRITING TO FILE")
        file.write('\n'.join(lines))

def simple_write_to_file(filename: str, text: str):
    # Ensure 'console_logs' directory exists and append output to the specified file
    logs_dir = 'console_logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    file_path = os.path.join(logs_dir, filename)
    with open(file_path, 'a') as file:
        # Formatting the start and end of the text
        formatted_text = "\n=== START OF TEXT =============================================\n" + \
                         text + \
                         "\n=== END OF TEXT ===============================================\n\n"
        file.write(formatted_text)



def write_class_to_console_logs(instance, filename):
    # Ensure 'console_logs' directory exists in the current working directory
    logs_dir = os.path.join(os.getcwd(), 'console_logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Specify the file path within the 'console_logs' directory
    file_path = os.path.join(logs_dir, filename)
    
    # Create a dictionary to hold class data
    class_data = {}
    # Iterate over all attributes of the class instance
    for attribute in dir(instance):
        # Filter out private and built-in attributes/methods
        if not attribute.startswith('_') and not callable(getattr(instance, attribute)):
            # Add attribute and its value to the class_data dictionary
            value = getattr(instance, attribute)
            # For non-simple data types (like lists or dicts), convert to a string for better readability
            if isinstance(value, (list, dict, datetime.datetime)):
                value = json.dumps(value, default=str, indent=2)
            class_data[attribute] = value
    
    # Formatting the class data as a string
    formatted_data = "\n".join(f"{key}: {value}" for key, value in class_data.items())
    
    # Writing the formatted data to the file within 'console_logs' directory
    with open(file_path, 'w') as file:
        file.write(formatted_data)

def save_gpt_prompt_to_file(filename: str, params: dict, prompt: str):
    # Ensure 'console_logs' directory exists and append output to the specified file
    logs_dir = 'console_logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    file_path = os.path.join(logs_dir, filename)
    with open(file_path, 'a') as file:
        # Convert parameters dictionary to a string
        params_str = "\n".join(f"{key}: {value}" for key, value in params.items())
        
        # Formatting the start and end of the text with parameters and prompt
        formatted_text = "\n=== START OF PARAMETERS ========================================\n" + \
                         params_str + \
                         "\n=== END OF PARAMETERS ==========================================\n" + \
                         "\n=== START OF PROMPT ============================================\n" + \
                         prompt + \
                         "\n=== END OF PROMPT ==============================================\n\n"
        file.write(formatted_text)