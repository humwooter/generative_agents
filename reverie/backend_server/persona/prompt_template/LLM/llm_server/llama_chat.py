from queue import Queue, Empty
import subprocess
import os
import threading

# Dictionary of chats, where the key is the UUID of the chat
chat_interfaces = {}

def kill_chat(chat_uuid):
    """Used for deleting the chat."""
    if chat_uuid in chat_interfaces:
        chat_interfaces[chat_uuid].end_chat()  # Properly terminate and clean up the chat
        del chat_interfaces[chat_uuid]
    else:
        print(f"No chat with UUID {chat_uuid} found.")

def run_ggml_model(model_path, userInput, chat_uuid, prompt_intro):
    """Run the GGML model with given inputs."""
    if chat_uuid not in chat_interfaces:  # Create new chat
        print("Creating new chat session")
        chat_interfaces[chat_uuid] = ChatInterface(chat_uuid=chat_uuid, prompt_intro=prompt_intro)
    chat_interfaces[chat_uuid].set_model_path(model_path)
    response = chat_interfaces[chat_uuid].chat(userInput)
    kill_chat(chat_uuid)  # Kill chat immediately after getting a response
    return response

class ChatInterface:
    def __init__(self, chat_uuid, prompt_intro) -> None:
        self.process = None
        self.q = Queue()
        self.chat_uuid = chat_uuid
        self.prompt_intro = prompt_intro  # Introduction prompt for the chat
        self.setup_chat_history()

    def setup_chat_history(self):
        """Setup chat history file and directory."""
        script_dir = os.path.dirname(os.path.realpath(__file__))
        history_dir = os.path.join(script_dir, 'Chat Histories')
        self.history_file = os.path.join(history_dir, f"{self.chat_uuid}_history.txt")
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

    def end_chat(self):
        """Properly terminates the chat process and cleans up."""
        if self.process:
            self.process.terminate()
        if os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                f.truncate(0)  # Clear the history file

    def set_model_path(self, model_path):
        """Configure and start the model subprocess."""
        # Navigate to the model directory and start the model subprocess
        target_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
        os.chdir(target_dir)
        # absolute_path_prompt_file = self.create_prompt_file()
        
        self.cmd = ['./main', '-m', model_path, '-n', '2048', '--repeat_penalty', '1.0', '--color', '-i', '-r', 'User:', '-f', absolute_path_prompt_file]
        self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=0, universal_newlines=True, encoding='utf-8')
        threading.Thread(target=self.read_output, args=(self.process.stdout, self.q), daemon=True).start()

    # def create_prompt_file(self):
    #     """Generate and write the prompt to a file."""
    #     # Prepare the prompt file content with intro and chat history
    #     prompt_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Prompts', f'prompt_{self.chat_uuid}.txt')
    #     os.makedirs(os.path.dirname(prompt_file_path), exist_ok=True)
    #     with open(prompt_file_path, 'w', encoding='utf-8') as f:
    #         f.write(self.prompt_intro)
    #     return prompt_file_path

    def read_output(self, pipe, q):
        """Read output from the subprocess and put it into the queue."""
        buffer = ""
        while True:
            line = pipe.readline()
            if line:
                buffer += line
                if "#END#" in buffer:
                    q.put(buffer.split("#END#")[0].strip())
                    break  # End reading after finding the delimiter

    def chat(self, userInput):
        """Send input to the model subprocess and wait for the response."""
        if not self.process:
            return "Model not initialized. Please select model file."
        
        self.process.stdin.write(f"User: {userInput}\n")
        self.process.stdin.flush()

        try:
            response = self.q.get(timeout=120)  # Wait for the response with a timeout
        except Empty:
            response = "Timeout or no response received."
        
        return response