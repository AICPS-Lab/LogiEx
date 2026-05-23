import os
import json
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler, OpenAI
import time
import sys
import os
import backend.utils as utils
import ssl
import httpx


ASSISTANT_ID = os.getenv('OPENAI_ASSISTANT_ID', 'asst_gk6c0sIuoYXTRqpuLEPp8m77')

# Cached client and assistant to avoid re-creating on every request
_cached_client = None
_cached_assistant = None
_cached_api_key = None

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    @override
    def on_message_done(self, message, client) -> None:
        # print a citation to the file searched
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f"[{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")

        print(message_content.value)
        print("\n".join(citations))


def llm_init(model, api_key=None, assistant=None):
    ssl_context = ssl.create_default_context()
    try:
        ssl_context.options |= ssl.OP_NO_TLSv1
        ssl_context.options |= ssl.OP_NO_TLSv1_1
    except AttributeError:
        pass
    http_client = httpx.Client(verify=ssl_context)

    client = OpenAI(api_key=api_key, http_client=http_client)

    # client = OpenAI(api_key=api_key)
    assistant = client.beta.assistants.update(
        ASSISTANT_ID)
    print("Assistant:", assistant.name)
    print("Assistant Tools:", assistant.tools)
    thread = client.beta.threads.create()
    return client, assistant, thread


def llm_init_wo_thread(model, api_key=None, assistant=None, thread_id=None):
    global _cached_client, _cached_assistant, _cached_api_key

    if _cached_client is not None and _cached_api_key == api_key:
        client = _cached_client
        assistant = _cached_assistant
    else:
        ssl_context = ssl.create_default_context()
        try:
            ssl_context.options |= ssl.OP_NO_TLSv1
            ssl_context.options |= ssl.OP_NO_TLSv1_1
        except AttributeError:
            pass
        http_client = httpx.Client(verify=ssl_context)

        client = OpenAI(api_key=api_key, http_client=http_client)
        assistant = client.beta.assistants.update(ASSISTANT_ID)
        _cached_client = client
        _cached_assistant = assistant
        _cached_api_key = api_key
        print("Assistant:", assistant.name)
        print("Assistant Tools:", assistant.tools)

    print("Thread ID:", thread_id)
    thread = client.beta.threads.retrieve(thread_id=thread_id)
    return client, assistant, thread


def get_mcts_context(context_text, max_length):
    """Read MCTS search tree info from a json file"""
    return context_text[:max_length]


def load_queries(start_index, end_index):
    query_file_path = "backend/queries/transit_queries_completed.txt"
    query_list = []
    with open(query_file_path, 'r') as file:
        for line in file:
            query_list.append(line.strip())
    return query_list[start_index:end_index+1]


def load_tree_file_openai(openai_client, index):
    file_name = "exp_test_{}.json".format(index)
    search_tree_file = os.path.join("backend/data/transit", file_name)
    # print("Uploading file", file_name)

    vector_store = openai_client.beta.vector_stores.create(
        name="Search Tree Reference"
    )
    uploaded_file = openai_client.files.create(
        file=open(search_tree_file, "rb"),
        purpose="assistants"
    )
    
    vector_store_file = openai_client.beta.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id
    )

    assistant = openai_client.beta.assistants.update(
        assistant_id=ASSISTANT_ID,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    print("Uploaded file ID:", uploaded_file.id)
    print("Vector_store ID:", vector_store.id)
    return vector_store.id, uploaded_file.id


def test_one_query(repeat_num, client, assistant, query):
    """Send a single query to gpt and get a response."""
    response_logs = []
    
    for i in range(repeat_num):

        thread = client.beta.threads.create()

        message = client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=query
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        utils.wait_on_run(run, client, thread)
        
        messages = client.beta.threads.messages.list(
            thread_id=thread.id, order="asc", after=message.id
        )
        
        model_response = messages.data[0].content[0].text.value
        # utils.print_break_string(model_response)
        response_logs.append(model_response)

    return response_logs
    

def test_one_round(args, client, query):
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    run = utils.wait_on_run(run, thread)

    messages = client.beta.threads.messages.list(
        thread_id=thread.id, order="asc", after=message.id
    )

    response = client.chat.completions.create(
        model=args.backend,
        messages=messages,
        temperature=args.temperature
    )
    
    conversation_history = messages
    
    model_response = response.choices[0].message.content
    print(f"AI: {model_response}")

    conversation_history.append({"role": "assistant", "content": model_response})

    def add_to_history(user_query, bot_response):
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": bot_response})

    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break
        
        conversation_history.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=conversation_history,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            
            model_response = response.choices[0].message.content
            print(f"AI: {model_response}") 

            add_to_history(user_input, model_response)

        except Exception as e:
            print(e)
            return ""



if __name__ == '__main__':
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    file = client.files.create(
        file=open(
            "data/language_models_are_unsupervised_multitask_learners.pdf",
            "rb",
        ),
        purpose="assistants",
    )

    assistant = client.beta.assistants.create(
        name="ExplainableMCTS-Helper",
        instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
        model="gpt-4-1106-preview")

    assistant = client.beta.assistants.update(
        ASSISTANT_ID,
        tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],
        file_ids=[file.id],)

    test_one_query()