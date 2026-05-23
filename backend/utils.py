
import time
from termcolor import colored
from openai import OpenAI



def print_break_string(text, max_length=75):
    lines = text.split('\n')
    formatted_text = []
    
    for line in lines:
        if len(line) > max_length:
            words = line.split(' ')
            new_line = ''
            temp_line = ''
            for word in words:
                if len(temp_line) + len(word) < max_length:
                    temp_line += word + ' '
                else:
                    formatted_text.append(temp_line.strip())
                    temp_line = word + ' '
            formatted_text.append(temp_line.strip())
        else:
            formatted_text.append(line)

    for line in formatted_text:
        print(line)


def pretty_print(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    print("# Messages\n")
    for message in messages:
        if message.role == "system":
            print(colored(f"system: {message.content[0].text.value}\n", role_to_color[message.role]))
        elif message.role == "user":
            print(colored(f"user: {message.content[0].text.value}\n", role_to_color[message.role]))
        elif message.role == "assistant":
            print(colored(f"assistant: {message.content[0].text.value}\n", role_to_color[message.role]))
        elif message.role == "function":
            print(colored(f"function ({message.name}): {message['content']}\n", role_to_color[message.role]))


def get_response(client, thread):
    """Get the latest response from GPT."""
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
    for message in messages: 
        if message.role == "assistant":
            return message.content[0].text.value


def wait_on_run(run, client, thread):
    """Wait for assistant to finish running."""
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve( thread_id=thread.id, run_id=run.id )
        time.sleep(0.3)
    return run


def submit_message(client, assistant_id, thread, user_message):
    """Submit a message to GPT."""
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message)
    
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id)


def create_thread_and_run(client, assistant_id, user_input, thread=None):
    """Create a thread and submit message."""
    """You do not need to create a new thread everytime."""
    if not thread:
        thread = client.beta.threads.create()
    run = submit_message(client, assistant_id, thread, user_input)
    return thread, run
        

def submit_file(file_path, client: OpenAI, sim_mode):
    with open(file_path, 'rb') as file:
        upload_file = client.files.create(file=file, purpose='assistants')
    return upload_file.id


def delete_file(client, vector_store_id, file_id):
    print(vector_store_id, file_id)
    deleted_vector_store_file = client.beta.vector_stores.files.delete(
        vector_store_id=vector_store_id,
        file_id=file_id
    )
    print("Removed file ID:", deleted_vector_store_file)