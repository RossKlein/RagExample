
import os
import openai
import numpy as np
import pandas as pd
import json
from pprint import pprint

f = open("key.txt", "r")

private_key = f.read()
f.close()
openai.api_key = private_key
messages = []

from llama_index.core import SimpleDirectoryReader

def cosine_similarity(A, B):
    return np.dot(A,B)## openai vectors are already normalized!

def embed_file(filename):
    cur_embeddings = None
    try:
        cur_embeddings = pd.read_parquet("database.parquet")
        if(filename in cur_embeddings['file'].tolist()):
            return
    except:
        pass

    documents = SimpleDirectoryReader(
        input_files=[filename]
    ).load_data()

    ################# get document###############
    documents = "\n\n".join([doc.text for doc in documents])

    ##########tokenize document to 256 tokens###################
    TOKENS = 256
    tokenize = documents.split()
    split_by = int(len(tokenize)/TOKENS)
    if len(tokenize)%TOKENS < TOKENS/2:
        split_by +=1
    print(split_by)################ split document into roughly equal chunks
    chunk_size = int(len(tokenize)/split_by)
    sublists = [tokenize[i * chunk_size:(i + 1) * chunk_size] for i in range(split_by - 1)]
    sublists.append(tokenize[(split_by - 1) * chunk_size:])


    ################# retrieve embeddings for each split chunk#################

    sublists = [" ".join(sub) for sub in sublists]
    embeddings = pd.DataFrame(openai.embeddings.create(input = sublists, model="text-embedding-3-small").data)
    embeddings = pd.DataFrame(embeddings[0].tolist())[1]
    df = pd.DataFrame({'text': sublists, 'embedding': embeddings, 'file': filename})
    if cur_embeddings is not None:
        cur_embeddings.append(df)
        cur_embeddings.to_parquet("database.parquet")
    else:
        df.to_parquet("database.parquet")
        
    print("done making embeddings")


def ask(question):

    # question = "can you tell me what this document is about?"
    Q_NUMBER = 4
    promptstr =     (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Rephrase the search query in "+str(Q_NUMBER)+" different ways, one on each line, "
    "related to the following input query. only respond with the "+str(Q_NUMBER)+" queries:\n")
    promptjson = [{"role": "system",  "content": promptstr},
                    {"role": "user", "content":     "Query: "+question+"\n Queries:\n"}]
    print(promptjson)
    embedding_questions = openai.chat.completions.create(model = "gpt-4-turbo-preview", messages= promptjson, temperature=0.1).choices[0].message.content
    embedding_questions = embedding_questions.split("\n")
    question_embedding = pd.DataFrame(openai.embeddings.create(input = embedding_questions, model = "text-embedding-3-small").data)
    embeddings = pd.DataFrame(question_embedding[0].tolist())[1]

    df = pd.DataFrame()
    try:
        df = pd.read_parquet("database.parquet")
    except:
        print("no database file please read a document first using the embed_file() function")
        return
    context = ""
    for q in embeddings:
        df['similarity'] = pd.DataFrame(df.embedding.tolist()).dot(pd.Series(q))
        df = df.sort_values("similarity", ascending=False)
        print(df)
        context += "\n\n".join(df.head(2)['text'])+"\n"

    messages.append(
        {"role": "system", "content": "You are an informative assistant who is meant to best answer the "
        + "users questions given the following excerpts from document(s) and previous chats and excerpts,"
        + "though this might be the first chat. Please provide a detailed answer."
        + "Remember that these are excerpts from a large dataset: \n" + context})
    messages.append({"role": "user", "content": question})
    pprint(messages, compact=True)
    response = openai.chat.completions.create(model="gpt-4-turbo-preview", messages = messages, temperature=0.1)
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )
    print("\n\n\n")
    print(response.choices[0].message.content)
ask("can you tell me what this document is about?")

