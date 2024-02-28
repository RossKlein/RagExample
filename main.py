
import os
import openai
import numpy as np
import pandas as pd
import json
from pprint import pprint
from transformers import BertTokenizer, BertModel
import random
import torch
import re
########### Set up BERT model to use###############
random_seed = 96
random.seed(random_seed)
 
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to('cuda')
####################################################

f = open("key.txt", "r")

private_key = f.read()
f.close()
openai.api_key = private_key
messages = []

from llama_index.core import SimpleDirectoryReader

def BERT_embeddings(datalist):
        ################ retrieve embeddings using BERT for each split chunk##################################
    batch_size = 200
    bert = None
    for i in range(0,len(datalist), batch_size):
        encoding = tokenizer.batch_encode_plus(
        datalist[i:i+batch_size],  # List of input texts
        padding=True,              
        truncation=True,           
        return_tensors='pt',    
        add_special_tokens=True
        )
        encoding.to('cuda')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state  # This contains the embeddings
        if bert is not None: 
            bert = np.concatenate([bert, word_embeddings.mean(dim=1).cpu().numpy()], axis = 0)
        else:
            bert = word_embeddings.mean(dim=1).cpu().numpy()
    return bert
    
    
def embed_file(filename):
    cur_embeddings = None
    try:
        cur_embeddings = pd.read_parquet("database.parquet")
        if(filename in cur_embeddings['file'].tolist()):
            print("file already in the database")
            return
    except:
        pass
    print("loading file...")
    documents = SimpleDirectoryReader(
        input_files=[filename]
    ).load_data()
    print("starting tokenize")
    ################# get document###############
    documents = "\n\n".join([doc.text for doc in documents])
    ##########tokenize document to x tokens################### thee is a way to parallelize with numpy
    TOKENS = 256
    documents = re.sub(r'\.{3,}', '~', documents)###
    tokenize = documents.split()
    split_by = int(len(tokenize)/TOKENS)
    if len(tokenize)%TOKENS < TOKENS/2:
        split_by +=1
    print(split_by)################ split document into roughly equal chunks
    chunk_size = int(len(tokenize)/split_by)    
    
    sublists = [tokenize[i * chunk_size:(i + 1) * chunk_size] for i in range(split_by - 1)]
    sublists.append(tokenize[(split_by - 1) * chunk_size:])
    
    sublists = [" ".join(sub) for sub in sublists]

    global bert
    bert = BERT_embeddings(sublists)
    bert = pd.DataFrame(bert).apply(lambda row: row.values, axis=1).to_frame(name='embeddings')
    bert = pd.DataFrame({"text": sublists,"embeddings": bert.embeddings,"file":filename})
    if cur_embeddings is not None:
        cur_embeddings = pd.concat([cur_embeddings, bert])
        cur_embeddings.to_parquet("database.parquet")
    else:
        bert.to_parquet("database.parquet")
    print("done making embeddings")    


def ask(question):

    # question = "can you tell me what this document is about?"
    global context_rankings
    context_rankings = pd.DataFrame()
    Q_NUMBER = 3
    promptstr_cat_synonym =     (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Rephrase the search query in "+str(Q_NUMBER)+" different ways, one on each line, "
    "related to the following input query. Also, you are meant to preface each generated query with "
    "a category which best describes the topic and at the end, append a list of important synonyms."
    "When picking a category and synonyms keep in mind how the question is asked is not important but the specific topics of the question are. "
    "only respond like this: "
    "(category): (question) -(synonym list)"
    "(category): (question) -(synonym list)"
    "(category): (question) -(synonym list)"
    "(category): (question) -(synonym list)")
    
    promptstr_cat =     (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Rephrase the search query in "+str(Q_NUMBER)+" different ways, one on each line, "
    "related to the following input query. Also, you are meant to preface each generated query with "
    "a category which best describes the topic and at the end."
    "When picking a category keep in mind how the question is asked is not important but the specific topics of the question are. "
    "only respond like this: "
    "(category): (question)"
    "(category): (question)"
    "(category): (question)"
    "(category): (question)"
    "(category): (question)")
    
    promptstr =     (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. The purpose is to use an embedding model to find the most similar text excerpts to these questions"
    "Rephrase the search query in "+str(Q_NUMBER)+" different ways, one on each line, "
    "related to the following input query."
    "only respond like this: "
    "(question)"
    "(question)"
    "...")
    promptjson = [{"role": "system",  "content": promptstr},
                    {"role": "user", "content":     "Query: "+question+"\n Queries:\n"}]
    embedding_questions = openai.chat.completions.create(model = "gpt-4-turbo-preview", messages= promptjson, temperature=0.1).choices[0].message.content
    embedding_questions = re.split(r'\n+', embedding_questions)
    embedding_questions.append(question)
    for q in embedding_questions:
        print(q + "\n")
    question_embedding = pd.DataFrame(BERT_embeddings(embedding_questions)).transpose()

    try:
        df = pd.read_parquet("database.parquet")
    except:
        print("no database file please read a document first using the embed_file() function")
        return
    context = ""
    for i in question_embedding.columns:
        df['similarity'] = pd.DataFrame(df.embeddings.tolist()).dot(pd.Series(question_embedding[i]))
        df = df.sort_values("similarity", ascending=False)
        context += "\n\n".join(df.head(3)['text'])+"\n"
        ####strangely, reranking the context gives WORSE
        # context_rankings = pd.concat([context_rankings, df.head(10)])
        print(df)
    # context_rankings = context_rankings.drop_duplicates(subset="text",keep="first").sort_values("similarity", ascending=False)
    # context += "\n\n".join(df.head(8)['text'])+"\n"
    print(context)
    messages.append(
        {"role": "system", "content": "You are an informative assistant who is meant to best answer the "
        + "users questions given the following excerpts from document(s) and previous chats and excerpts."
        + " Please provide a detailed answer. And do your best to make inferences based on the excerpts provided"
        + ", remember that these are excerpts from a large dataset: \n" + context})
    messages.append({"role": "user", "content": question})
    # pprint(messages, compact=True)
    response = openai.chat.completions.create(model="gpt-4-turbo-preview", messages = messages, temperature=0.1)
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )
    print("\n\n\n")
    print(response.choices[0].message.content)
def clear():
    messages = []
ask("can you tell me what this document is about?")

