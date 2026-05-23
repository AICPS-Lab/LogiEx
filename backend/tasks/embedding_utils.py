# reference: https://platform.openai.com/docs/guides/embeddings/use-cases. 
from openai import OpenAI 
import pandas as pd 
import numpy as np
import tiktoken 
import os
from scipy import spatial



def strings_ranked_by_relatedness(client, query: str, data_ref: list, 
                                  relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
                                  top_n: int = 100, 
                                  model="text-embedding-3-small") -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=model, input=query)
    query_embedding = query_embedding_response.data[0].embedding
    
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"])) for i, row in enumerate(data_ref)
    ]

    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]



def query_message(query: str, df: pd.DataFrame, model: str, token_budget: int) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below document on the paratransit service to answer the subsequent question. If the answer cannot be found in the document, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction

    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (num_tokens(message + next_article + question, model=model) > token_budget):
            break
        else:
            message += next_article
    return message + question



def ask(client, query: str, df: pd.DataFrame = None, model: str = None,
    token_budget: int = 4096 - 500, print_message: bool = False) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message



def num_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)
    

def get_embedding(client, text, model="text-embedding-3-small"):
    """Get direct embeddings from text."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def get_reduced_embedding(client, text, model="text-embedding-3-small"):
    """Get reduced embeddings from text."""
    text = text.replace("\n", " ")
    create_embed = client.embeddings.create(input = [text], model=model) 
    cut_dim = create_embed.data[0].embedding[:256]
    norm_dim = normalize_l2(cut_dim)
    return norm_dim



if __name__ == '__main__':
    file_path = "../data/transit_knowledge.md"
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embedding_model = "text-embedding-3-small"
    embedding_encoding = "cl100k_base"
    max_tokens = 8000

    knowledge_encoding = [get_embedding(client, x, model=embedding_model) for x in content]
    content_lib = []
    for i in range(len(content)):
        content_lib_entry = {}
        content_lib_entry["text"] = content[i]
        content_lib_entry["embedding"] = knowledge_encoding[i]
        content_lib.append(content_lib_entry)
    
    kb_queries_path = "../queries/knowledge_base_queries.txt"
    with open(kb_queries_path, 'r') as file: 
        kb_queries = []
        for line in file:
            kb_queries.append(line.strip())
    
    for query in kb_queries: 
        print(query)
        strings, relatednesses = strings_ranked_by_relatedness(client, query, content_lib, top_n=3)
        for string, relatedness in zip(strings, relatednesses):
            print(f"{relatedness=:.3f}")
            print(string)
    
    exit()