"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: run_gpt_prompt.py
Description: Defines all run gpt prompt functions. These functions directly
interface with the safe_generate_response function.
"""

import re
import datetime
import sys
import ast
import re

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *
from persona.prompt_template.LLM.LocalLLM import LocalLLM
from datetime import datetime, timedelta


def get_random_alphanumeric(i=6, j=6): 
  """
  Returns a random alpha numeric strength that has the length of somewhere
  between i and j. 

  INPUT: 
    i: min_range for the length
    j: max_range for the length
  OUTPUT: 
    an alpha numeric str with the length of somewhere between i and j.
  """
  k = random.randint(i, j)
  x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
  return x


##############################################################################
# CHAPTER 1: Run GPT Prompt
##############################################################################

def run_llm_prompt_wake_up_hour(persona, test_input=None, verbose=False, model_name="gpt-4", server_url="http://localhost:8000"):
    """
    Given the persona, returns an integer that indicates the hour when the
    persona wakes up.

    INPUT:
      persona: The Persona class instance
    OUTPUT:
      integer for the wake up hour.
    """
    def create_prompt_input(persona, test_input=None):
        if test_input: return test_input
        prompt_input = [persona.scratch.get_str_iss(),
                        persona.scratch.get_str_lifestyle(),
                        persona.scratch.get_str_firstname()]
        return prompt_input

    def __func_clean_up(llm_response, prompt=""):
        cr = int(llm_response.strip().lower().split("am")[0])
        return cr

    def __func_validate(llm_response, prompt=""):
        try: __func_clean_up(llm_response, prompt="")
        except: return False
        return True

    def get_fail_safe():
        fs = 8
        return fs

    # Initialize LocalLLM with the model name and server URL
    local_llm = LocalLLM(model_name=model_name, server_url=server_url)

    prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
    prompt_input = create_prompt_input(persona, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    # Use LocalLLM to generate a response
    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = fail_safe

    if verbose:
        print_run_prompts(prompt_template, persona, {"model_name": model_name, "server_url": server_url}, prompt_input, prompt, output)

    return output, [output, prompt, {"model_name": model_name, "server_url": server_url}, prompt_input, fail_safe]



def run_gpt_prompt_daily_plan(persona, wake_up_hour, test_input=None, verbose=False, model_name="gpt-4", server_url="http://localhost:8000"):
    """
    Basically the long term planning that spans a day. Returns a list of actions
    that the persona will take today. Usually comes in the following form: 
    'wake up and complete the morning routine at 6:00 am', 
    'eat breakfast at 7:00 am',.. 
    Note that the actions come without a period. 

    INPUT: 
      persona: The Persona class instance 
    OUTPUT: 
      a list of daily actions in broad strokes.
    """
    def create_prompt_input(persona, wake_up_hour, test_input=None):
        if test_input: return test_input
        prompt_input = [persona.scratch.get_str_iss(),
                        persona.scratch.get_str_lifestyle(),
                        persona.scratch.get_str_curr_date_str(),
                        persona.scratch.get_str_firstname(),
                        f"{str(wake_up_hour)}:00 am"]
        return prompt_input

    def __func_clean_up(llm_response, prompt=""):
        cr = []
        _cr = llm_response.split(")")
        for i in _cr: 
            if i[-1].isdigit(): 
                i = i[:-1].strip()
                if i[-1] == "." or i[-1] == ",": 
                    cr += [i[:-1].strip()]
        return cr

    def __func_validate(llm_response, prompt=""): 
        try: __func_clean_up(llm_response, prompt="")
        except: 
            return False
        return True

    def get_fail_safe(): 
        fs = ['wake up and complete the morning routine at 6:00 am', 
              'eat breakfast at 7:00 am', 
              'read a book from 8:00 am to 12:00 pm', 
              'have lunch at 12:00 pm', 
              'take a nap from 1:00 pm to 4:00 pm', 
              'relax and watch TV from 7:00 pm to 8:00 pm', 
              'go to bed at 11:00 pm'] 
        return fs

    # Initialize LocalLLM with the model name and server URL
    local_llm = LocalLLM(model_name=model_name, server_url=server_url)

    prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
    prompt = " ".join(prompt_input)  # Assuming generate_prompt joins the input list into a single string prompt
    fail_safe = get_fail_safe()

    # Use LocalLLM to generate a response
    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = fail_safe

    if verbose:
        # Adjust logging as needed
        print(f"Prompt: {prompt}")
        print(f"Response: {output}")

    return output


def run_llm_prompt_generate_hourly_schedule(persona, 
                                            curr_hour_str,
                                            p_f_ds_hourly_org, 
                                            hour_str,
                                            intermission2=None,
                                            test_input=None, 
                                            verbose=False,
                                            model_name="gpt-4",
                                            server_url="http://localhost:8000"): 
    def create_prompt_input(persona, 
                            curr_hour_str, 
                            p_f_ds_hourly_org,
                            hour_str,
                            intermission2=None,
                            test_input=None): 
        if test_input: return test_input
        schedule_format = ""
        for i in hour_str: 
            schedule_format += f"[{persona.scratch.get_str_curr_date_str()} -- {i}]"
            schedule_format += f" Activity: [Fill in]\n"
        schedule_format = schedule_format[:-1]

        intermission_str = f"Here the originally intended hourly breakdown of"
        intermission_str += f" {persona.scratch.get_str_firstname()}'s schedule today: "
        for count, i in enumerate(persona.scratch.daily_req): 
            intermission_str += f"{str(count+1)}) {i}, "
        intermission_str = intermission_str[:-2]

        prior_schedule = ""
        if p_f_ds_hourly_org: 
            prior_schedule = "\n"
            for count, i in enumerate(p_f_ds_hourly_org): 
                prior_schedule += f"[(ID:{get_random_alphanumeric()})" 
                prior_schedule += f" {persona.scratch.get_str_curr_date_str()} --"
                prior_schedule += f" {hour_str[count]}] Activity:"
                prior_schedule += f" {persona.scratch.get_str_firstname()}"
                prior_schedule += f" is {i}\n"

        prompt_ending = f"[(ID:{get_random_alphanumeric()})"
        prompt_ending += f" {persona.scratch.get_str_curr_date_str()}"
        prompt_ending += f" -- {curr_hour_str}] Activity:"
        prompt_ending += f" {persona.scratch.get_str_firstname()} is"

        if intermission2: 
            intermission2 = f"\n{intermission2}"

        prompt_input = []
        prompt_input += [schedule_format]
        prompt_input += [persona.scratch.get_str_iss()]

        prompt_input += [prior_schedule + "\n"]
        prompt_input += [intermission_str]
        if intermission2: 
            prompt_input += [intermission2]
        else: 
            prompt_input += [""]
        prompt_input += [prompt_ending]

        return prompt_input

    def __func_clean_up(llm_response, prompt=""):
        cr = llm_response.strip()
        if cr[-1] == ".":
            cr = cr[:-1]
        return cr

    def __func_validate(llm_response, prompt=""): 
        try: __func_clean_up(llm_response, prompt="")
        except: return False
        return True

    def get_fail_safe(): 
        fs = "asleep"
        return fs

    # Initialize LocalLLM with the model name and server URL
    local_llm = LocalLLM(model_name=model_name, server_url=server_url)

    prompt_template = "persona/prompt_template/v2/generate_hourly_schedule_v2.txt"
    prompt_input = create_prompt_input(persona, 
                                       curr_hour_str, 
                                       p_f_ds_hourly_org,
                                       hour_str, 
                                       intermission2,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    # Use LocalLLM to generate a response
    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = fail_safe

    if verbose:
        print_run_prompts(prompt_template, persona, {"model_name": model_name, "server_url": server_url}, prompt_input, prompt, output)

    return output, [output, prompt, {"model_name": model_name, "server_url": server_url}, prompt_input, fail_safe]




def run_llm_prompt_task_decomp(persona, task, duration, test_input=None, verbose=False, model_name="your_model_name", server_url="http://your_server_url"):
    def create_prompt_input(persona, task, duration, test_input=None):
        curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
        all_indices = [curr_f_org_index]
        if curr_f_org_index+1 <= len(persona.scratch.f_daily_schedule_hourly_org): 
            all_indices += [curr_f_org_index+1]
        if curr_f_org_index+2 <= len(persona.scratch.f_daily_schedule_hourly_org): 
            all_indices += [curr_f_org_index+2]

        curr_time_range = ""
        summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. From '
        for index in all_indices: 
            if index < len(persona.scratch.f_daily_schedule_hourly_org): 
                start_min = sum(duration for _, duration in persona.scratch.f_daily_schedule_hourly_org[:index])
                end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
                start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") + datetime.timedelta(minutes=start_min)).strftime("%H:%M%p")
                end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") + datetime.timedelta(minutes=end_min)).strftime("%H:%M%p")
                summ_str += f"{start_time} ~ {end_time}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
                if curr_f_org_index+1 == index:
                    curr_time_range = f'{start_time} ~ {end_time}'
        summ_str = summ_str[:-2] + "."
        
        prompt_input = [
            persona.scratch.get_str_iss(),
            summ_str,
            persona.scratch.get_str_firstname(),
            task,
            curr_time_range,
            duration,
            persona.scratch.get_str_firstname()
        ]
        return prompt_input

    def __func_clean_up(llm_response, prompt=""):
        temp = [i.strip() for i in llm_response.split("\n")]
        cr = []
        for count, i in enumerate(temp): 
            if count != 0: 
                task_info = " ".join(j.strip() for j in i.split(" ")[3:])
                task, duration_str = task_info.split("(duration in minutes:")
                duration = int(duration_str.split(",")[0].strip())
                if task[-1] == ".": 
                    task = task[:-1]
                cr.append([task, duration])
        return cr

    def __func_validate(llm_response, prompt=""): 
        try: 
            cleaned_response = __func_clean_up(llm_response, prompt)
            if not cleaned_response:
                return False
        except Exception as e:
            return False
        return True

    def get_fail_safe(): 
        return ["asleep"]

    local_llm = LocalLLM(model_name=model_name, server_url=server_url)

    prompt_input = create_prompt_input(persona, task, duration, test_input)
    prompt = " ".join(prompt_input)  # Adjust this to match your prompt generation logic
    fail_safe = get_fail_safe()

    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response, prompt):
            output = __func_clean_up(llm_response, prompt)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = fail_safe

    if verbose:
        print("Prompt:", prompt)
        print("LLM Response:", llm_response)
        print("Output:", output)

    return output, [output, prompt, {"model_name": model_name, "server_url": server_url}, prompt_input, fail_safe]



def run_llm_prompt_action_sector(action_description, persona, maze, test_input=None, verbose=False, model_name="your_model_name", server_url="http://your_server_url"):
    """Generates an action sector based on the description, persona, and maze configuration using a LocalLLM."""
    
    def create_prompt_input(action_description, persona, maze, test_input=None):
        """Constructs the input prompt from action description, persona details, and maze configuration."""
        act_world = maze.access_tile(persona.scratch.curr_tile)['world']
        
        # compile prompt inputs from various sources
        prompt_input = [
            persona.scratch.get_str_name(),
            persona.scratch.living_area.split(":")[1],
            persona.s_mem.get_str_accessible_sector_arenas(f"{act_world}:{persona.scratch.living_area.split(':')[1]}"),
            persona.scratch.get_str_name(),
            maze.access_tile(persona.scratch.curr_tile)['sector'],
            persona.s_mem.get_str_accessible_sector_arenas(f"{act_world}:{maze.access_tile(persona.scratch.curr_tile)['sector']}"),
        ]
        
        # include daily plan if available
        daily_plan = persona.scratch.get_str_daily_plan_req()
        prompt_input.append(f"\n{daily_plan}" if daily_plan != "" else "")
        
        # filter accessible sectors based on specific conditions
        accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
        accessible_sectors = [sector for sector in accessible_sector_str.split(", ") if "'s house" not in sector or persona.scratch.last_name in sector]
        prompt_input.append(", ".join(accessible_sectors))
        
        # handle action description variations
        action_parts = action_description.split("(") if "(" in action_description else [action_description, action_description]
        action_description_1 = action_parts[0].strip()
        action_description_2 = action_parts[-1][:-1].strip() if len(action_parts) > 1 else action_description_1
        prompt_input.extend([persona.scratch.get_str_name(), action_description_1, action_description_2, persona.scratch.get_str_name()])
        
        return prompt_input

    prompt_input = create_prompt_input(action_description, persona, maze, test_input)
    prompt = " ".join(prompt_input)  # adjust formatting as needed for your llm
    
    local_llm = LocalLLM(model_name=model_name, server_url=server_url)
    
    try:
        # call the llm and process the response
        llm_response = local_llm._call(prompt)
        output = llm_response  # process response as needed for your application
    except Exception as e:
        if verbose:
            print(f"error calling localllm: {e}")
        output = "error"  # consider a failsafe response or error handling strategy

    if verbose:
        # log prompt and response for debugging
        print("prompt:", prompt)
        print("response:", output)

    return output



def run_llm_prompt_action_arena(action_description, persona, maze, act_world, act_sector, test_input=None, verbose=False):
    """
    Generates action arena based on the description, persona details, and the specified world and sector.
    Assumes LocalLLM class is already provided for interacting with a local LLM model.
    """

    def create_prompt_input(action_description, persona, act_world, act_sector):
        """Constructs the input prompt from action description, persona details, world, and sector."""
        prompt_input = [
            persona.scratch.get_str_name(),  # persona's name
            act_sector,  # current sector
        ]

        # Retrieve accessible arenas in the sector and filter based on specific conditions
        accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(f"{act_world}:{act_sector}")
        accessible_arenas = [arena for arena in accessible_arena_str.split(", ") if "'s room" not in arena or persona.scratch.last_name in arena]
        accessible_arena_str = ", ".join(accessible_arenas)
        prompt_input.append(accessible_arena_str)

        # Handle action description variations
        action_description_1, action_description_2 = (action_description.split("(")[0].strip(), action_description.split("(")[-1][:-1]) if "(" in action_description else (action_description, action_description)
        prompt_input.extend([persona.scratch.get_str_name(), action_description_1, action_description_2, persona.scratch.get_str_name(), act_sector, accessible_arena_str])

        return prompt_input

    def __func_clean_up(llm_response):
        """Cleans up the LLM response for use."""
        cleaned_response = llm_response.split("}")[0]  # assuming response format needs trimming
        return cleaned_response

    def __func_validate(llm_response):
        """Validates the LLM response to ensure it meets expected criteria."""
        if len(llm_response.strip()) < 1 or "}" not in llm_response or "," in llm_response:
            return False
        return True

    def get_fail_safe():
        """Provides a fail-safe response in case of validation failure or errors."""
        return "kitchen"  # example of a fail-safe response

    # Create input prompt based on the action description and persona details
    prompt_input = create_prompt_input(action_description, persona, act_world, act_sector)
    prompt = " ".join(prompt_input)  # Convert list of input components into a single string prompt

    # Initialize the LocalLLM with specified model and server URL
    local_llm = LocalLLM(model_name="your_model_name", server_url="http://your_server_url")

    try:
        # Generate response using the local LLM
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = get_fail_safe()
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = get_fail_safe()

    if verbose:
        # Log prompts and output for debugging purposes
        print("Prompt:", prompt)
        print("LLM Response:", output)

    return output, [output, prompt, {"model_name": "your_model_name", "server_url": "http://your_server_url"}, prompt_input, get_fail_safe()]




def run_llm_prompt_action_game_object(action_description, persona, maze, temp_address, test_input=None, verbose=False, model_name="your_model_name", server_url="http://your_server_url"):
    # create the input prompt based on the action description, persona, and the temporary address
    def create_prompt_input(action_description, persona, temp_address, test_input=None):
        prompt_input = []
        if "(" in action_description:
            action_description = action_description.split("(")[-1][:-1]
        prompt_input += [action_description]
        prompt_input += [persona.s_mem.get_str_accessible_arena_game_objects(temp_address)]
        return prompt_input

    # validate the response from the llm to ensure it's not empty
    def __func_validate(llm_response, prompt=""):
        if len(llm_response.strip()) < 1:
            return False
        return True

    # clean up the response from the llm by stripping unnecessary whitespace
    def __func_clean_up(llm_response, prompt=""):
        cleaned_response = llm_response.strip()
        return cleaned_response

    # define a fail-safe response in case the llm fails to generate a valid response
    def get_fail_safe():
        fs = "bed"
        return fs

    # initialize the LocalLLM with the specified model name and server URL
    local_llm = LocalLLM(model_name=model_name, server_url=server_url)

    # generate the prompt input
    prompt_input = create_prompt_input(action_description, persona, temp_address, test_input)
    prompt = " ".join(prompt_input)  # join the prompt input into a single string

    # define a fail-safe response
    fail_safe = get_fail_safe()

    # use the LocalLLM to generate a response
    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = fail_safe

    # ensure the output is one of the accessible game objects
    accessible_objects = [i.strip() for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")]
    if output not in accessible_objects:
        output = random.choice(accessible_objects)

    # log the process if verbose mode is enabled
    if verbose:
        print(f"Prompt: {prompt}")
        print(f"LLM Response: {output}")

    return output, [output, prompt, {"model_name": model_name, "server_url": server_url}, prompt_input, fail_safe]




def run_llm_prompt_pronunciation(action_description, persona, verbose=False, model_name="your_model_name", server_url="http://your_server_url"):
    # create the input prompt based on the action description
    def create_prompt_input(action_description):
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
        prompt_input = [action_description]
        return prompt_input

    # clean up the llm response to ensure it's concise
    def __func_clean_up(llm_response, prompt=""):
        cr = llm_response.strip()
        if len(cr) > 3:
            cr = cr[:3]  # limit the response to 3 characters
        return cr

    # validate the llm response to ensure it's not empty
    def __func_validate(llm_response, prompt=""):
        try:
            __func_clean_up(llm_response, prompt="")
            if len(llm_response) == 0:
                return False
        except:
            return False
        return True

    # define a fail-safe response in case the llm fails to generate a valid response
    def get_fail_safe():
        fs = "😋"  # default fail-safe emoji
        return fs

    # initialize the LocalLLM with the specified model name and server URL
    local_llm = LocalLLM(model_name=model_name, server_url=server_url)

    # generate the prompt input
    prompt_input = create_prompt_input(action_description)
    prompt = " ".join(prompt_input)  # join the prompt input into a single string

    # define a fail-safe response
    fail_safe = get_fail_safe()

    # use the LocalLLM to generate a response
    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = fail_safe

    # log the process if verbose mode is enabled
    if verbose:
        print(f"Prompt: {prompt}")
        print(f"LLM Response: {output}")

    return output, [output, prompt, {"model_name": model_name, "server_url": server_url}, prompt_input, fail_safe]  


def run_llm_prompt_event_triple(action_description, persona, local_llm, verbose=False):
    # Define a function to create the input prompt based on the action description and persona
    def create_prompt_input(action_description, persona):
        # Extract the core action description if it's enclosed in parentheses
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
        # Construct the prompt input list with the persona's name and the action description
        prompt_input = [persona.name, action_description, persona.name]
        return prompt_input

    # Define a function to clean up the response from the LLM
    def __func_clean_up(llm_response, prompt=""):
        # Strip leading and trailing whitespace and split the response into components
        cr = llm_response.strip()
        cr = [i.strip() for i in cr.split(")")[0].split(",")]
        return cr

    # Define a function to validate the LLM's response
    def __func_validate(llm_response, prompt=""):
        try:
            # Clean up the response and check if it has exactly 2 components
            llm_response = __func_clean_up(llm_response, prompt="")
            if len(llm_response) != 2:
                return False
        except:
            return False
        return True

    # Define a function to provide a fail-safe response
    def get_fail_safe(persona):
        # Return a default event triple indicating the persona is idle
        fs = (persona.name, "is", "idle")
        return fs

    # Generate the prompt input using the provided action description and persona
    prompt_input = create_prompt_input(action_description, persona)
    # Join the prompt input components into a single string prompt
    prompt = " ".join(prompt_input)

    # Define a fail-safe response
    fail_safe = get_fail_safe(persona)

    # Use the LocalLLM instance to generate a response
    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = fail_safe

    # If verbose mode is enabled, print the prompt and the output for debugging
    if verbose:
        print("Prompt:", prompt)
        print("LLM Response:", llm_response)
        print("Output:", output)

    # Return the output along with additional information for debugging or further processing
    return output, [output, prompt, {"local_llm": local_llm}, prompt_input, fail_safe]



def run_llm_prompt_act_obj_desc(act_game_object, act_desp, persona, local_llm, verbose=False):
    # create the input prompt based on the action description, persona, and the game object
    def create_prompt_input(act_game_object, act_desp, persona):
        prompt_input = [act_game_object, persona.name, act_desp, act_game_object, act_game_object]
        return prompt_input

    # clean up the llm response by trimming and removing the final period if present
    def __func_clean_up(llm_response, prompt=""):
        cr = llm_response.strip()
        if cr[-1] == ".": cr = cr[:-1]
        return cr

    # validate the llm response to ensure it's not empty and can be cleaned up
    def __func_validate(llm_response, prompt=""):
        try:
            llm_response = __func_clean_up(llm_response, prompt="")
        except:
            return False
        return True

    # define a fail-safe response in case the llm fails to generate a valid response
    def get_fail_safe(act_game_object):
        fs = f"{act_game_object} is idle"
        return fs

    # generate the prompt input
    prompt_input = create_prompt_input(act_game_object, act_desp, persona)
    # join the prompt input into a single string
    prompt = " ".join(prompt_input)
    # define a fail-safe response
    fail_safe = get_fail_safe(act_game_object)

    # use the LocalLLM instance to generate a response
    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = fail_safe

    # return the output along with additional information for debugging or further processing
    return output, [output, prompt, {"local_llm": local_llm}, prompt_input, fail_safe]



def run_llm_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona, local_llm, verbose=False):
    # define a function to create the input prompt based on the game object and its description
    def create_prompt_input(act_game_object, act_obj_desc):
        prompt_input = [act_game_object, act_obj_desc, act_game_object]
        return prompt_input

    # define a function to clean up the llm response, extracting the event triple components
    def __func_clean_up(llm_response, prompt=""):
        cr = llm_response.strip()
        cr = [i.strip() for i in cr.split(")")[0].split(",")]
        return cr

    # define a function to validate the llm response, ensuring it contains exactly two components
    def __func_validate(llm_response, prompt=""):
        try:
            llm_response = __func_clean_up(llm_response, prompt="")
            if len(llm_response) != 2:
                return False
        except:
            return False
        return True

    # define a function to provide a fail-safe response in case the llm fails to generate a valid response
    def get_fail_safe(act_game_object):
        fs = (act_game_object, "is", "idle")
        return fs

    # create the input prompt using the provided game object and description
    prompt_input = create_prompt_input(act_game_object, act_obj_desc)
    # join the prompt input components into a single string prompt
    prompt = " ".join(prompt_input)

    # define a fail-safe response
    fail_safe = get_fail_safe(act_game_object)

    # use the LocalLLM instance to generate a response
    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"Error generating LLM response: {e}")
        output = fail_safe

    # format the output as an event triple
    output = (act_game_object, output[0], output[1])

    # if verbose mode is enabled, print the prompt and the output for debugging
    if verbose:
        print("Prompt:", prompt)
        print("LLM Response:", llm_response)
        print("Output:", output)

    # return the output along with additional information for debugging or further processing
    return output, [output, prompt, {"local_llm": local_llm}, prompt_input, fail_safe]


# Revised function to use LocalLLM for generating new decomposed schedules
def run_llm_prompt_new_decomp_schedule(persona, main_act_dur, truncated_act_dur, start_time_hour, end_time_hour, inserted_act, inserted_act_dur, test_input=None, verbose=False):
    # function to create the input prompt for the llm based on the given parameters
    def create_prompt_input(persona, main_act_dur, truncated_act_dur, start_time_hour, end_time_hour, inserted_act, inserted_act_dur, test_input=None):
        # formatting persona details and schedule times
        persona_name = persona.name
        start_hour_str = start_time_hour.strftime("%H:%M %p")
        end_hour_str = end_time_hour.strftime("%H:%M %p")

        # constructing the original and new plans based on activities and their durations
        original_plan, new_plan_init = "", ""
        for_time = start_time_hour
        for i in main_act_dur:
            original_plan += f'{for_time.strftime("%H:%M")} ~ {(for_time + timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0] + "\n"
            for_time += timedelta(minutes=int(i[1]))

        for_time = start_time_hour
        for count, i in enumerate(truncated_act_dur):
            new_plan_init += f'{for_time.strftime("%H:%M")} ~ {(for_time + timedelta(minutes=int(i[1]))).strftime("%H:%M")} -- ' + i[0] + "\n"
            if count < len(truncated_act_dur) - 1:
                for_time += timedelta(minutes=int(i[1]))
        new_plan_init += (for_time + timedelta(minutes=int(i[1]))).strftime("%H:%M") + " ~"

        # compiling all parts into a single prompt input list
        prompt_input = [persona_name, start_hour_str, end_hour_str, original_plan, persona_name, inserted_act, inserted_act_dur, persona_name, start_hour_str, end_time_hour, end_time_hour, new_plan_init]
        return prompt_input

    # function to process and clean up the llm response
    def func_clean_up(llm_response, prompt=""):
        # extracting and formatting the new schedule from the llm response
        new_schedule = prompt + " " + llm_response.strip().split("The revised schedule:")[-1].strip().split("\n")
        ret_temp = [i.split(" -- ") for i in new_schedule]

        ret = []
        for time_str, action in ret_temp:
            start_time, end_time = time_str.split(" ~ ")
            delta = datetime.strptime(end_time.strip(), "%H:%M") - datetime.strptime(start_time.strip(), "%H:%M")
            delta_min = max(int(delta.total_seconds() / 60), 0)
            ret.append([action, delta_min])
        return ret

    # function to validate the llm response
    def func_validate(llm_response, prompt=""):
        try:
            llm_response = func_clean_up(llm_response, prompt)
            dur_sum = sum(dur for _, dur in llm_response)
            prompt_times = prompt.split("\n")[0].split("originally planned schedule from")[-1].strip()[:-1].split(" to ")
            prompt_start, prompt_end = [datetime.strptime(i.strip(), "%H:%M %p") for i in prompt_times]
            delta_min = int((prompt_end - prompt_start).total_seconds() / 60)
            return dur_sum == delta_min and all(isinstance(act, str) and isinstance(dur, int) for act, dur in llm_response)
        except:
            return False

    # setting up the LocalLLM instance
    local_llm = LocalLLM(model_name="your_model_name_here", server_url="http://localhost:8000")  # adjust model_name and server_url as needed

    # generating the prompt input
    prompt_input = create_prompt_input(persona, main_act_dur, truncated_act_dur, start_time_hour, end_time_hour, inserted_act, inserted_act_dur, test_input)
    prompt = "\n".join(prompt_input)  # converting the prompt input list into a string

    # generating the response from the LocalLLM
    llm_response = local_llm._call(prompt)  # using the _call method to get the response

    # validating and cleaning up the response
    if func_validate(llm_response, prompt):
        output = func_clean_up(llm_response, prompt)
    else:
        output = "Failed to generate valid response"

    if verbose:
        print("Prompt:", prompt)
        print("LLM Response:", llm_response)
        print("Output:", output)

    return output



# revised function to decide on initiating conversation using LocalLLM
def run_llm_prompt_decide_to_talk(persona, target_persona, retrieved, test_input=None, verbose=False):
    # create the input prompt for the llm
    def create_prompt_input(init_persona, target_persona, retrieved, test_input=None):
        # retrieve last chat details
        last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
        last_chatted_time, last_chat_about = "", ""
        if last_chat:
            last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
            last_chat_about = last_chat.description

        # compile context from events and thoughts
        context = ""
        for c_node in retrieved["events"]:
            curr_desc = c_node.description.split(" ")
            curr_desc[2:3] = ["was"]
            context += " ".join(curr_desc) + ". "
        for c_node in retrieved["thoughts"]:
            context += f"{c_node.description}. "

        # format current time and activity descriptions
        curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
        init_act_desc = init_persona.scratch.act_description.split("(")[-1][:-1] if "(" in init_act_desc else init_act_desc
        target_act_desc = target_persona.scratch.act_description.split("(")[-1][:-1] if "(" in target_act_desc else target_act_desc

        # describe current and target persona activities
        init_p_desc = f"{init_persona.name} is {'already ' if len(init_persona.scratch.planned_path) == 0 and 'waiting' not in init_act_desc else ''}{init_act_desc}"
        target_p_desc = f"{target_persona.name} is {'already ' if len(target_persona.scratch.planned_path) == 0 and 'waiting' not in target_act_desc else ''}{target_act_desc}"

        # compile all parts into a single prompt input list
        prompt_input = [context, curr_time, init_persona.name, target_persona.name, last_chatted_time, last_chat_about, init_p_desc, target_p_desc]
        return prompt_input

    # validate the llm response
    def func_validate(llm_response, prompt=""):
        return llm_response.strip().lower() in ["yes", "no"]

    # clean up the llm response
    def func_clean_up(llm_response, prompt=""):
        return llm_response.strip().lower()

    # define a fail-safe response
    def get_fail_safe():
        return "yes"

    # setting up the LocalLLM instance
    local_llm = LocalLLM(model_name="your_model_name_here", server_url="http://localhost:8000")  # adjust as needed

    # generating the prompt input and the prompt
    prompt_input = create_prompt_input(persona, target_persona, retrieved, test_input)
    prompt = "\n".join(prompt_input)  # converting list to string

    # generating the response from the LocalLLM
    llm_response = local_llm._call(prompt)  # using the _call method

    # validating and cleaning up the response
    output = func_clean_up(llm_response) if func_validate(llm_response) else get_fail_safe()

    if verbose:
        print("Prompt:", prompt)
        print("LLM Response:", llm_response)
        print("Output:", output)

    return output

# revised function to decide on a reaction using LocalLLM
def run_llm_prompt_decide_to_react(persona, target_persona, retrieved, test_input=None, verbose=False):
    # create the input prompt for the llm
    def create_prompt_input(init_persona, target_persona, retrieved, test_input=None):
        # compile context from events and thoughts
        context = ""
        for c_node in retrieved["events"]:
            curr_desc = " ".join(c_node.description.split(" ", 3)[:3] + ["was"] + c_node.description.split(" ", 3)[3:])
            context += f"{curr_desc}. "
        context += "\n"
        for c_node in retrieved["thoughts"]:
            context += f"{c_node.description}. "

        # format current time
        curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")

        # prepare activity descriptions and locations
        init_act_desc = init_persona.scratch.act_description.split("(")[-1][:-1] if "(" in init_act_desc else init_act_desc
        target_act_desc = target_persona.scratch.act_description.split("(")[-1][:-1] if "(" in target_act_desc else target_act_desc

        loc_init = init_persona.scratch.act_address.split(":")[-1] + " in " + init_persona.scratch.act_address.split(":")[-2] if ":" in init_persona.scratch.act_address else ""
        loc_target = target_persona.scratch.act_address.split(":")[-1] + " in " + target_persona.scratch.act_address.split(":")[-2] if ":" in target_persona.scratch.act_address else ""

        init_p_desc = f"{init_persona.name} is {'already ' if len(init_persona.scratch.planned_path) == 0 else 'on the way to '}{init_act_desc} at {loc_init}"
        target_p_desc = f"{target_persona.name} is {'already ' if len(target_persona.scratch.planned_path) == 0 else 'on the way to '}{target_act_desc} at {loc_target}"

        # explicitly constructing the prompt_input list
        prompt_input = []
        prompt_input += [context]  # adding context
        prompt_input += [curr_time]  # adding current time
        prompt_input += [init_p_desc]  # adding initiator's description
        prompt_input += [target_p_desc]  # adding target's description
        prompt_input += [init_persona.name]  # adding initiator's name
        prompt_input += [init_act_desc]  # adding initiator's activity description
        prompt_input += [target_persona.name]  # adding target's name
        prompt_input += [target_act_desc]  # adding target's activity description
        prompt_input += [init_act_desc]  # adding initiator's activity description again

        return prompt_input


    # validate the llm response
    def func_validate(llm_response, prompt=""):
        return llm_response.strip().lower() in ["3", "2", "1"]

    # clean up the llm response
    def func_clean_up(llm_response, prompt=""):
        return llm_response.strip().lower()

    # define a fail-safe response
    def get_fail_safe():
        return "3"

    # setting up the LocalLLM instance
    local_llm = LocalLLM(model_name="your_model_name_here", server_url="http://localhost:8000")  # adjust as needed

    # generating the prompt input and the prompt
    prompt_input = create_prompt_input(persona, target_persona, retrieved, test_input)
    prompt = "\n".join(prompt_input)  # converting list to string

    # generating the response from the LocalLLM
    llm_response = local_llm._call(prompt)  # using the _call method

    # validating and cleaning up the response
    output = func_clean_up(llm_response) if func_validate(llm_response) else get_fail_safe()

    if verbose:
        print("Prompt:", prompt)
        print("LLM Response:", llm_response)
        print("Output:", output)

    return output


def run_llm_prompt_create_conversation(persona, target_persona, curr_loc, local_llm, test_input=None, verbose=False):
    # import re

    # function to create the input prompt based on the current context and personas
    def create_prompt_input(init_persona, target_persona, curr_loc, test_input=None):
        prev_convo_insert = "\n"
        # check for previous conversations and format them for inclusion in the prompt
        if init_persona.a_mem.seq_chat:
            for i in init_persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((init_persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, they had the following conversation.\n'
                    for row in i.filling:
                        prev_convo_insert += f'{row[0]}: "{row[1]}"\n'
                    break
            if prev_convo_insert == "\n":
                prev_convo_insert = ""
            if init_persona.a_mem.seq_chat:
                if int((init_persona.scratch.curr_time - init_persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
                    prev_convo_insert = ""

        # retrieve and format thoughts relevant to the interaction
        init_persona_thought_nodes = init_persona.a_mem.retrieve_relevant_thoughts(target_persona.scratch.act_event[0],
                                                                                    target_persona.scratch.act_event[1],
                                                                                    target_persona.scratch.act_event[2])
        init_persona_thought = ""
        for i in init_persona_thought_nodes:
            init_persona_thought += f"-- {i.description}\n"

        target_persona_thought_nodes = target_persona.a_mem.retrieve_relevant_thoughts(init_persona.scratch.act_event[0],
                                                                                        init_persona.scratch.act_event[1],
                                                                                        init_persona.scratch.act_event[2])
        target_persona_thought = ""
        for i in target_persona_thought_nodes:
            target_persona_thought += f"-- {i.description}\n"

        # describe the current activities of both personas
        init_persona_curr_desc = f"{init_persona.name} is {'on the way to ' if init_persona.scratch.planned_path else ''}{init_persona.scratch.act_description}"
        target_persona_curr_desc = f"{target_persona.name} is {'on the way to ' if target_persona.scratch.planned_path else ''}{target_persona.scratch.act_description}"

        # compile all parts into a single prompt input list
        prompt_input = [init_persona.scratch.get_str_iss(), target_persona.scratch.get_str_iss(),
                        init_persona.name, target_persona.name, init_persona_thought,
                        target_persona.name, init_persona.name, target_persona_thought,
                        init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S"),
                        init_persona_curr_desc, target_persona_curr_desc, prev_convo_insert,
                        init_persona.name, target_persona.name, curr_loc["arena"], init_persona.name]
        return prompt_input

    # function to clean up the llm response, extracting conversation content
    def __func_clean_up(llm_response, prompt=""):
        llm_response = (prompt + llm_response).split("What would they talk about now?")[-1].strip()
        content = re.findall('"([^"]*)"', llm_response)
        speaker_order = [i.split(":")[0].strip() for i in llm_response.split("\n") if i.split(":")[0].strip()]
        return [[speaker, content[count]] for count, speaker in enumerate(speaker_order)]

    # function to validate the llm response
    def __func_validate(llm_response, prompt=""):
        try:
            __func_clean_up(llm_response, prompt)
            return True
        except:
            return False

    # define a fail-safe response in case the llm fails to generate a valid response
    def get_fail_safe(init_persona, target_persona):
        return [[init_persona.name, "Hi!"], [target_persona.name, "Hi!"]]

    # generate the prompt input
    prompt_input = create_prompt_input(persona, target_persona, curr_loc, test_input)
    # join the prompt input components into a single string prompt
    prompt = " ".join(prompt_input)

    # define a fail-safe response
    fail_safe = get_fail_safe(persona, target_persona)

    # use the LocalLLM instance to generate a response
    try:
        llm_response = local_llm._call(prompt)
        if __func_validate(llm_response):
            output = __func_clean_up(llm_response)
        else:
            output = fail_safe
    except Exception as e:
        if verbose:
            print(f"LLM call failed with error: {e}")
        output = fail_safe

    # if debug or verbose mode is enabled, print the run prompts for debugging purposes
    if verbose:
        # Assuming print_run_prompts is a custom function for logging, replaced with print for this context
        print("Prompt:", prompt)
        print("Output:", output)

    return output, [output, prompt, {"engine": local_llm.model_name}, prompt_input, fail_safe]



def run_gpt_prompt_summarize_conversation(persona, conversation, test_input=None, verbose=False): 
  def create_prompt_input(conversation, test_input=None): 
    convo_str = ""
    for row in conversation: 
      convo_str += f'{row[0]}: "{row[1]}"\n'

    prompt_input = [convo_str]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    ret = "conversing about " + gpt_response.strip()
    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "conversing with a housemate about morning greetings"


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    ret = "conversing about " + gpt_response.strip()
    return ret

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 


  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 11") ########
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt" ########
  prompt_input = create_prompt_input(conversation, test_input)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "conversing about what to eat for lunch" ########
  special_instruction = "The output must continue the sentence above by filling in the <fill in> tag. Don't start with 'this is a conversation about...' Just finish the sentence but do not miss any important details (including who are chatting)." ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

def run_gpt_prompt_extract_keywords(persona, description, test_input=None, verbose=False): 
  def create_prompt_input(description, test_input=None): 
    if "\n" in description: 
      description = description.replace("\n", " <LINE_BREAK> ")
    prompt_input = [description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    print ("???")
    print (gpt_response)
    gpt_response = gpt_response.strip().split("Emotive keywords:")
    factual = [i.strip() for i in gpt_response[0].split(",")]
    emotive = [i.strip() for i in gpt_response[1].split(",")]
    all_keywords = factual + emotive
    ret = []
    for i in all_keywords: 
      if i: 
        i = i.lower()
        if i[-1] == ".": 
          i = i[:-1]
        ret += [i]
    print (ret)
    return set(ret)

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return []

  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/get_keywords_v1.txt"
  prompt_input = create_prompt_input(description, test_input)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)


  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_keyword_to_thoughts(persona, keyword, concept_summary, test_input=None, verbose=False): 
  def create_prompt_input(persona, keyword, concept_summary, test_input=None): 
    prompt_input = [keyword, concept_summary, persona.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response.strip()
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return ""

  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 40, 
               "temperature": 0.7, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/keyword_to_thoughts_v1.txt"
  prompt_input = create_prompt_input(persona, keyword, concept_summary)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_convo_to_thoughts(persona, 
                                    init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None, verbose=False): 
  def create_prompt_input(init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None): 
    prompt_input = [init_persona_name,
                    target_persona_name,
                    convo_str,
                    init_persona_name,
                    fin_target]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response.strip()
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return ""

  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 40, 
               "temperature": 0.7, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/convo_to_thoughts_v1.txt"
  prompt_input = create_prompt_input(init_persona_name,  
                                    target_persona_name,
                                    convo_str,
                                    fin_target)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_llm_prompt_event_poignancy(persona, event_description, test_input=None, verbose=False):
    """
    Generates a poignancy score for an event description using a language model.
    
    Args:
        persona: The persona object containing metadata for the prompt generation.
        event_description (str): A description of the event to analyze.
        test_input: Optional parameter for testing purposes.
        verbose (bool): If True, prints additional debug information.
        
    Returns:
        tuple: A tuple containing the poignancy score and additional information for debugging.
    """
    
    def create_prompt_input(persona, event_description, test_input=None):
        """Creates the input for the prompt based on the persona and event description."""
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        event_description]
        return prompt_input
    
    def __func_clean_up(llm_response, prompt=""):
        """Cleans up the LLM response, attempting to parse it as an integer."""
        try:
            llm_response = int(llm_response.strip())
            return llm_response
        except ValueError:
            return get_fail_safe()

    def __func_validate(llm_response, prompt=""): 
        """Validates the LLM response by attempting to parse it as an integer."""
        try: 
            int(llm_response.strip())
            return True
        except ValueError:
            return False 

    def get_fail_safe(): 
        """Defines a fail-safe response in case of errors."""
        return 4

    # Debug print statement
    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 7") ########

    prompt_input = create_prompt_input(persona, event_description, test_input)
    prompt = generate_prompt(prompt_input, "persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt")
    
    # Model parameters, adjust as needed
    model_params = {
        "max_tokens": 15,
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None
    }
    
    output = generic_llm_request(prompt, model_params)
    cleaned_output = __func_clean_up(output, prompt)
    
    if verbose:
        print(cleaned_output)
    
    # Return type adjusted to match original functionality
    return cleaned_output, [cleaned_output, prompt, model_params, prompt_input, get_fail_safe()]


def run_gpt_prompt_thought_poignancy(persona, event_description, test_input=None, verbose=False): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = int(gpt_response.strip())
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return 4

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    gpt_response = int(gpt_response)
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 8") ########
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_thought_v1.txt" ########
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "5" ########
  special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10." ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  



def run_gpt_prompt_chat_poignancy(persona, event_description, test_input=None, verbose=False): 
  def create_prompt_input(persona, event_description, test_input=None): 
    prompt_input = [persona.scratch.name,
                    persona.scratch.get_str_iss(),
                    persona.scratch.name,
                    event_description]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = int(gpt_response.strip())
    return gpt_response

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return 4


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    gpt_response = int(gpt_response)
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 9") ########
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt" ########
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = "5" ########
  special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10." ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  



def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False): 
  def create_prompt_input(persona, statements, n, test_input=None): 
    prompt_input = [statements, str(n)]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = "1) " + gpt_response.strip()
    ret = []
    for i in gpt_response.split("\n"): 
      ret += [i.split(") ")[-1]]
    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(n): 
    return ["Who am I"] * n


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    ret = ast.literal_eval(gpt_response)
    return ret

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 


  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 12") ########
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt" ########
  prompt_input = create_prompt_input(persona, statements, n)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]' ########
  special_instruction = "Output must be a list of str." ########
  fail_safe = get_fail_safe(n) ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  # ChatGPT Plugin ===========================================================






  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 150, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/generate_focal_pt_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe(n)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]




  
def run_gpt_prompt_insight_and_guidance(persona, statements, n, test_input=None, verbose=False): 
  def create_prompt_input(persona, statements, n, test_input=None): 
    prompt_input = [statements, str(n)]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    gpt_response = "1. " + gpt_response.strip()
    ret = dict()
    for i in gpt_response.split("\n"): 
      row = i.split(". ")[-1]
      thought = row.split("(because of ")[0].strip()
      evi_raw = row.split("(because of ")[1].split(")")[0].strip()
      evi_raw = re.findall(r'\d+', evi_raw)
      evi_raw = [int(i.strip()) for i in evi_raw]
      ret[thought] = evi_raw
    return ret

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(n): 
    return ["I am hungry"] * n




  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 150, 
               "temperature": 0.5, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/insight_and_evidence_v1.txt"
  prompt_input = create_prompt_input(persona, statements, n)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe(n)
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]








def run_gpt_prompt_agent_chat_summarize_ideas(persona, target_persona, statements, curr_context, test_input=None, verbose=False): 
  def create_prompt_input(persona, target_persona, statements, curr_context, test_input=None): 
    prompt_input = [persona.scratch.get_str_curr_date_str(), curr_context, persona.scratch.currently, 
                    statements, persona.scratch.name, target_persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.split('"')[0].strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 17") ########
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_ideas_v1.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  



def run_gpt_prompt_agent_chat_summarize_relationship(persona, target_persona, statements, test_input=None, verbose=False): 
  def create_prompt_input(persona, target_persona, statements, test_input=None): 
    prompt_input = [statements, persona.scratch.name, target_persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.split('"')[0].strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 18") ########
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, statements)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  





def run_gpt_prompt_agent_chat(maze, persona, target_persona,
                               curr_context, 
                               init_summ_idea, 
                               target_summ_idea, test_input=None, verbose=False): 
  def create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea, test_input=None): 
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat: 
      for i in persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if persona.a_mem.seq_chat: 
      if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""
    print (prev_convo_insert)

    curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"
    

    prompt_input = [persona.scratch.currently, 
                    target_persona.scratch.currently, 
                    prev_convo_insert,
                    curr_context, 
                    curr_location,

                    persona.scratch.name,
                    init_summ_idea, 
                    persona.scratch.name,
                    target_persona.scratch.name,

                    target_persona.scratch.name,
                    target_summ_idea, 
                    target_persona.scratch.name,
                    persona.scratch.name,

                    persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    print (gpt_response)

    gpt_response = (prompt + gpt_response).split("Here is their conversation.")[-1].strip()
    content = re.findall('"([^"]*)"', gpt_response)

    speaker_order = []
    for i in gpt_response.split("\n"): 
      name = i.split(":")[0].strip() 
      if name: 
        speaker_order += [name]

    ret = []
    for count, speaker in enumerate(speaker_order): 
      ret += [[speaker, content[count]]]

    return ret



  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."




  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    # ret = ast.literal_eval(gpt_response)

    print ("a;dnfdap98fh4p9enf HEREE!!!")
    for row in gpt_response: 
      print (row)

    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): ############
    return True


  # print ("HERE JULY 23 -- ----- ") ########
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/agent_chat_v1.txt" ########
  prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = '[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]' ########
  special_instruction = 'The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"].' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  # print ("HERE END JULY 23 -- ----- ") ########
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_summarize_ideas(persona, statements, question, test_input=None, verbose=False): 
  def create_prompt_input(persona, statements, question, test_input=None): 
    prompt_input = [statements, persona.scratch.name, question]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.split('"')[0].strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 16") ########
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_ideas_v1.txt" ########
  prompt_input = create_prompt_input(persona, statements, question)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe is working on a project' ########
  special_instruction = 'The output should be a string that responds to the question.' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_generate_next_convo_line(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None, verbose=False): 
  def create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None): 
    prompt_input = [persona.scratch.name, 
                    persona.scratch.get_str_iss(),
                    persona.scratch.name, 
                    interlocutor_desc, 
                    prev_convo, 
                    persona.scratch.name,
                    retrieved_summary, 
                    persona.scratch.name,]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."

  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 250, 
               "temperature": 1, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/generate_next_convo_line_v1.txt"
  prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]






def run_gpt_prompt_generate_whisper_inner_thought(persona, whisper, test_input=None, verbose=False): 
  def create_prompt_input(persona, whisper, test_input=None): 
    prompt_input = [persona.scratch.name, whisper]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."

  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/whisper_inner_thought_v1.txt"
  prompt_input = create_prompt_input(persona, whisper)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_planning_thought_on_convo(persona, all_utt, test_input=None, verbose=False): 
  def create_prompt_input(persona, all_utt, test_input=None): 
    prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."

  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/planning_thought_on_convo_v1.txt"
  prompt_input = create_prompt_input(persona, all_utt)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def run_gpt_prompt_memo_on_convo(persona, all_utt, test_input=None, verbose=False): 
  def create_prompt_input(persona, all_utt, test_input=None): 
    prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
    return prompt_input
  
  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""): 
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."


  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response, prompt=""): ############
    return gpt_response.strip()

  def __chat_func_validate(gpt_response, prompt=""): ############
    try: 
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      return False 


  print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 15") ########
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 15, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v3_ChatGPT/memo_on_convo_v1.txt" ########
  prompt_input = create_prompt_input(persona, all_utt)  ########
  prompt = generate_prompt(prompt_input, prompt_template)
  example_output = 'Jane Doe was interesting to talk to.' ########
  special_instruction = 'The output should ONLY contain a string that summarizes anything interesting that the agent may have noticed' ########
  fail_safe = get_fail_safe() ########
  output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                          __chat_func_validate, __chat_func_clean_up, True)
  if output != False: 
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
  # ChatGPT Plugin ===========================================================

  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  prompt_template = "persona/prompt_template/v2/memo_on_convo_v1.txt"
  prompt_input = create_prompt_input(persona, all_utt)
  prompt = generate_prompt(prompt_input, prompt_template)

  fail_safe = get_fail_safe()
  output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)

  if debug or verbose: 
    print_run_prompts(prompt_template, persona, gpt_param, 
                      prompt_input, prompt, output)
  
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]




def run_gpt_generate_safety_score(persona, comment, test_input=None, verbose=False): 
  def create_prompt_input(comment, test_input=None):
    prompt_input = [comment]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = json.loads(gpt_response)
    return gpt_response["output"]

  def __chat_func_validate(gpt_response, prompt=""): 
    try: 
      fields = ["output"]
      response = json.loads(gpt_response)
      for field in fields: 
        if field not in response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  print ("11")
  prompt_template = "persona/prompt_template/safety/anthromorphosization_v1.txt" 
  prompt_input = create_prompt_input(comment) 
  print ("22")
  prompt = generate_prompt(prompt_input, prompt_template)
  print (prompt)
  fail_safe = get_fail_safe() 
  output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  print (output)
  
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



def extract_first_json_dict(data_str):
    # Find the first occurrence of a JSON object within the string
    start_idx = data_str.find('{')
    end_idx = data_str.find('}', start_idx) + 1

    # Check if both start and end indices were found
    if start_idx == -1 or end_idx == 0:
        return None

    # Extract the first JSON dictionary
    json_str = data_str[start_idx:end_idx]

    try:
        # Attempt to parse the JSON data
        json_dict = json.loads(json_str)
        return json_dict
    except json.JSONDecodeError:
        # If parsing fails, return None
        return None


def run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None, verbose=False): 
  def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None):
    persona = init_persona
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat: 
      for i in persona.a_mem.seq_chat: 
        if i.object == target_persona.scratch.name: 
          v1 = int((persona.scratch.curr_time - i.created).total_seconds()/60)
          prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
          break
    if prev_convo_insert == "\n": 
      prev_convo_insert = ""
    if persona.a_mem.seq_chat: 
      if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
        prev_convo_insert = ""
    print (prev_convo_insert)

    curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    curr_arena= f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"

    retrieved_str = ""
    for key, vals in retrieved.items(): 
      for v in vals: 
        retrieved_str += f"- {v.description}\n"


    convo_str = ""
    for i in curr_chat:
      convo_str += ": ".join(i) + "\n"
    if convo_str == "": 
      convo_str = "[The conversation has not started yet -- start it!]"

    init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
    prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
      curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
      convo_str, init_persona.scratch.name, target_persona.scratch.name,
      init_persona.scratch.name, init_persona.scratch.name,
      init_persona.scratch.name
      ]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = extract_first_json_dict(gpt_response)

    cleaned_dict = dict()
    cleaned = []
    for key, val in gpt_response.items(): 
      cleaned += [val]
    cleaned_dict["utterance"] = cleaned[0]
    cleaned_dict["end"] = True
    if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
      cleaned_dict["end"] = False

    return cleaned_dict

  def __chat_func_validate(gpt_response, prompt=""): 
    print ("ugh...")
    try: 
      # print ("debug 1")
      # print (gpt_response)
      # print ("debug 2")

      print (extract_first_json_dict(gpt_response))
      # print ("debug 3")

      return True
    except:
      return False 

  def get_fail_safe():
    cleaned_dict = dict()
    cleaned_dict["utterance"] = "..."
    cleaned_dict["end"] = False
    return cleaned_dict

  print ("11")
  prompt_template = "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt" 
  prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat) 
  print ("22")
  prompt = generate_prompt(prompt_input, prompt_template)
  print (prompt)
  fail_safe = get_fail_safe() 
  output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  print (output)
  
  gpt_param = {"engine": "gpt-3.5-turbo-instruct", "max_tokens": 50, 
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]



















