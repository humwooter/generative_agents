from queue import Queue, Empty
import subprocess
import os
import threading
# from transformers import AutoTokenizer, AutoModel



#dictionary of chats, where the key is the uuid of the chat
chat_interfaces = {}

#used for deleting the chat
def kill_chat(chat_uuid):
    if chat_uuid in chat_interfaces:
        chat_interfaces[chat_uuid].process.terminate()
        del chat_interfaces[chat_uuid]
    else:
        print(f"No chat with UUID {chat_uuid} found.")

def run_ggml_model(model_path, userInput, chat_uuid):
    if chat_uuid not in chat_interfaces: #create new chat
        print("creating new chat session")
        chat_interfaces[chat_uuid] = ChatInterface(chat_uuid=chat_uuid)
        chat_interfaces[chat_uuid].set_model_path(model_path)
    elif chat_interfaces[chat_uuid] == None:
        print("chat interface exists in dictionary but subprocess doesn't exist")

    return chat_interfaces[chat_uuid].chat(userInput)



class ChatInterface:
    def __init__(self, chat_uuid) -> None:
        self.process = None
        self.q = Queue()
        self.t = None #what's this
        self.chat_uuid = chat_uuid

        script_dir = os.path.dirname(os.path.realpath(__file__))
        history_dir = os.path.join(script_dir, 'Chat Histories')
        self.history_file = os.path.join(history_dir, f"{chat_uuid}_history.txt")
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True) #create chat history file if it doesn't exist

        #load/initialize chat history
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.history = f.read().splitlines()
        else:
            self.history = []

        
    def end_chat(self):
        self.process.terminate()
        if os.path.exists(self.history_file):
            open(self.history_file, 'w').close()
        with open(self.history_file, 'r', encoding='utf-8') as f:
            for line in self.history:
                print(line, file=f)
    
    def create_prompt_file(self, num_words=100):
        intro_prompt = "This is the conversation history between a User and an AI Assistant named Bob. Bob is helpful, friendly, and tries to have natural conversations. The #END# delimiter marks the end of a response"
        # intro_prompt = "This is a chat with Bob, an AI assistant. Assume the role of Bob and continue the conversation." 

        # Create history_file if it doesn't exist already
        if not os.path.exists(self.history_file):
            open(self.history_file, 'w').close()
        
        with open(self.history_file, 'r', encoding='utf-8') as f:
            history_content = f.readlines()


        # Reverse the history_content to start counting words from the end
        reversed_content = history_content[::-1]
        word_count = 0
        selected_content = []

        # Iterate over reversed_content and stop when word_count is as close to num_words as possible
        for line in reversed_content:
            words_in_line = line.split()
            if word_count + len(words_in_line) > num_words:
                break
            word_count += len(words_in_line)
            selected_content.append(line)

        # Reverse selected_content to maintain original order and join to form a string
        history_content = ' '.join(selected_content[::-1])
                
        full_prompt = f"{intro_prompt}\n{history_content}"

        # Create the Prompts directory if it doesn't exist
        prompts_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Prompts')
        os.makedirs(prompts_dir, exist_ok=True)

        # Write the prompt to a file in the Prompts directory
        prompt_file_path = os.path.join(prompts_dir, f'prompt_{self.chat_uuid}.txt')
        with open(prompt_file_path, 'w', encoding='utf-8') as f:
            f.write(full_prompt)

        # Get the absolute path to the prompt file
        script_dir = os.path.dirname(os.path.realpath(__file__))
        prompts_dir = os.path.join(script_dir, 'Prompts')
        prompt_file_path = os.path.join(prompts_dir, f'prompt_{self.chat_uuid}.txt')
        return prompt_file_path
            
    def set_model_path(self, model_path):
        #paths 
        model_dir = os.path.dirname(model_path)
        parent_dir = os.path.dirname(model_dir)
        target_dir = os.path.dirname(parent_dir)
        os.chdir(target_dir)

        absolute_path_prompt_file = self.create_prompt_file()
        
        self.cmd = ['./main', '-m', model_path, '-n', '2048', '--repeat_penalty', '1.0', '--color', '-i', '-r', 'User:', '-f', absolute_path_prompt_file]
        self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=0, universal_newlines=True, encoding='utf-8')
        self.t = threading.Thread(target=self.read_output, args=(self.process.stdout, self.q))
        
        # Sets the thread 'self.t' as a daemon thread, allowing the main program to exit even if this thread is still running.
        self.t.daemon = True
        self.t.start()

    def read_output(self, pipe, q):
        buffer = ""
        while True:
            line = pipe.readline()
            if line:
                buffer += line
                if "#END#" in buffer:
                    split_output = buffer.split("#END#")
                    #collect all output except for end delimeter
                    for partial_output in split_output[:-1]:
                        q.put(partial_output.strip())
                    buffer = split_output[-1] #set buffer to whatever came after end delimeter (should be empty)

    
    def chat(self, userInput, timeout_seconds=120):
        if self.process is None:
            return "Model not initialized. Please select model file"
        
        if not userInput.strip():
            return "Please provide more information or clarify your question."
        
        # Add end delimiter to userInput and save it to history
        # userInput += "#END#"
        
        # Save userInput to history
        self.history.append(f"{userInput}")
        if not os.path.exists(self.history_file):
            open(self.history_file, 'w').close()

        #append new chats to the chat history
        with open(self.history_file, 'a', encoding='utf-8') as f:
            print(f"{userInput}", file=f)


        self.process.stdin.write(f"User: {userInput}\n")
        self.process.stdin.flush()


        # Clear the queue before listening for new output
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except Empty:
                break
    
        output_lines = []

        while True:
            try:
                line = self.q.get(timeout=timeout_seconds)
                if "Bob:" in line:
                    output_lines.append(line)
                    break
            except Empty:
                print("Process timed out.")
                # self.end_chat()
                return "Timeout"
            
        output = ''.join(output_lines)
        output += '#END#'
        self.history.append(output)

        with open(self.history_file, 'a', encoding='utf-8') as f:
            print(output, file=f)
        return output
    