import math
import json
import os
import voyageai
import re
import asyncio
import pandas as pd
from pinecone import Pinecone

def chunks(arr, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    length = len(arr)
    n = math.ceil(length/batch_size)
    chunks = [arr[i*batch_size:(i+1)*batch_size] for i in range(n)]
    return chunks
def panda_chunks(df, batch_size=100):
    global chunk
    """A helper function to break an iterable into chunks of size batch_size."""
    length = len(df)
    n = math.ceil(length/batch_size)
    
    df = pd.DataFrame({"id": df.index,"values": df.embeddings})
    df.id = df.id.astype(str)
    chunk = []
    for i in range(n):
        chunk.append(df.iloc[i*batch_size:(i+1)*batch_size])

    return chunk
        




##get openai secret key##############
##openai key
f = open("key.txt", "r")

private_key = f.read()
f.close()
##voyage key
f = open("voyagekey.txt", "r")

voyage_key = f.read()
f.close()
##pinecone key
f = open("pineconekey.txt", "r")

pinecone_key = f.read()
f.close()

#####################################
# voyage finance
# https://docs.voyageai.com/docs/embeddings
# email contact@voyageai.com
# voyage-finance-2
#####################################

## API KEYS
vo = voyageai.AsyncClient(voyage_key)
pc = Pinecone(api_key=pinecone_key)
os.environ["PINECONE_API_KEY"] = pinecone_key


## import document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
async def embed_helper(data_chunked):
    embedding = pd.DataFrame()
    voyage = []
    tasks = set()
    async with asyncio.TaskGroup() as tg:
        for chunk in data_chunked:
            task = tg.create_task(vo.embed(chunk, model="voyage-large-2"))
            tasks.add(task)
    for result in tasks:
        voyage.append(pd.DataFrame({'embeddings': result.result().embeddings}))
        
    return voyage
    
def embed_file(filename):
    global data
    global embeddings
    global documents
    global text
    cur_embeddings = None
    print("loading document...")
    loader = PyMuPDFLoader(filename)
    documents = loader.load()
    for i in range(len(documents)):
        temp = re.sub(r'\.{3,}', '~', documents[i].page_content)
        temp = re.sub(r'\{', '(', temp)
        documents[i].page_content = re.sub(r'\}', ')', temp)
        
        
    print("done loading...")
##tokenize document
    tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
    
    chunk_size = 4000
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    data = pd.DataFrame(docs)
    source_page = pd.DataFrame(pd.DataFrame(data[1].tolist())[1].tolist())
    text = pd.DataFrame(data[0].tolist())[1]
    print("done tokenizing...")
## embed document
    # embeddings = openai.embeddings.create(input = docs, model="text-embedding-3-small").data
    data_chunked = chunks(text.tolist(), int(120000/chunk_size))
    embeddings = pd.DataFrame()
    embeddings = asyncio.run(embed_helper(data_chunked))
    embeddings = pd.concat(embeddings)
    
    embeddings['text'] = text
    embeddings.index = text.index
    embeddings = pd.merge(embeddings, source_page, left_index=True, right_index=True)
    print("finished embedding")
    
    
## vector store document

    # vectorstore= PineconeVectorStore.from_texts(embeddings.text.tolist(), embeddings.embeddings.tolist(), index_name="main")
    # print(vectorstore.similarity_search("what is this document about", k=3))
    with pc.Index('main', pool_threads=30) as index:
        async_results = [
            index.upsert(vectors=json.loads(chunk.to_json(orient='records')), async_req=True) 
            for chunk in panda_chunks(embeddings, batch_size=100)
            
        ]
        for async_result in async_results:
            try:
                print(async_result.get())
            except Exception as e:
                print(e)
    embeddings.to_parquet("database.parquet")

## prompt engineering
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
#document retrieval
store = {}
session_id = 1


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def ask(query = "What is this document about"):
    global answer
    global context
    embeddings = pd.read_parquet("database.parquet")
    chat = ChatOpenAI(model="gpt-4-turbo-preview",temperature=0, openai_api_key=private_key)
    
##agent implementation???
    question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are meant to distinguish if"
         + "this query is asking for more information or if it is a follow up question."
         + "yes means it is asking for more information, no means it is a follow up."
         + "You should favor no if the query isn't referring to the document."
         + "You should favor yes if the query specifically requests for information from the document."
         + "If yes, optionally, you can provide a different"
         + "query that you think better captures the context of the question. You can either leave the optional query blank or provide a query by replacing OPTIONAL_QUERY."
         + "Pick one of the following answer choices: \n"
         + "YES OPTIONAL_QUERY"
         + "NO"),
        ("user", "last message: {history}"),
        ("user", "{input}")
    ]
    )
    question_runnable = question_prompt | chat
    
    history_length = len(get_session_history(session_id).messages)
    answer = ""
    if(history_length == 0):
        answer = question_runnable.invoke({"history": "", "input": query}).content
    else: 
        answer = question_runnable.invoke({"history": str(get_session_history(session_id).messages[history_length-1]), "input": query}).content
    context = ""
    print("####################")
    print(answer)
    print("####################")
    answer = answer.split(maxsplit= 1)
    if(answer[0].upper() == "YES"):
        if(len(answer) > 1):
            index = pc.Index("main")
            q_embedding = asyncio.run(embed_helper([answer[1]]))[0].embeddings.tolist()[0]
            returnval = index.query(vector=q_embedding,top_k=8,include_values=False)
            returnval = pd.DataFrame(returnval.to_dict()['matches'])
            final = pd.DataFrame(embeddings.iloc[returnval.id])
            returnval.index = final.index
            final['similarity'] = returnval['score']
            for t in final.text:
                # print(t)
                # print("\n")
                context += t + "\n\n"
        else:
            index = pc.Index("main")
            q_embedding = asyncio.run(embed_helper([query]))[0].embeddings.tolist()[0]
            returnval = index.query(vector=q_embedding,top_k=5,include_values=False)
            returnval = pd.DataFrame(returnval.to_dict()['matches'])
            final = pd.DataFrame(embeddings.iloc[returnval.id])
            returnval.index = final.index
            final['similarity'] = returnval['score']
            for t in final.text:
                # print(t)
                # print("\n")
                context += t + "\n\n"
        
            
        
##Chat prompting

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",  "You are an informative assistant who is meant to best answer the "
        + "users questions given the following excerpts from document(s) and previous chats and excerpts."
        + "Please provide a detailed answer and learn from the excerpts even if they do not provide the exact context needed."
        + "If the provided context is especially off topic, please mention that the excerpts do not provide relevant information."
        + "Remember to answer the users question primarily: \n"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    runnable = prompt | chat
    
    with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    )
    print(with_message_history.invoke(
    {"input": "Context: \n" + context + "\n\n" + "Question: " + query},
    config={"configurable": {"session_id": session_id}},).content)
    
def clear():
    session_id += 1
    
##evaluation??