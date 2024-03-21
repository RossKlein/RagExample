Run with https://pypi.org/project/ipython/  ipython is an interactive python module that allows you to run program functions with custom parameters while the program stays live

```
ipython -i main.py
```
make sure to make a `key.txt` file to store your openai private key


First, create an embedding on a document or documents:
```
embed_file("yourfile.pdf")
```
Ask it a question with:
```
ask("what is this document about?")
```


You can embed as many files as you want. If you want to make a new file database, delete or rename `database.parquet` 

# NOTE
There are 3 different versions of this program. The most accessible is the master branch. Other versions require extra setup. the CUDA branch is implemented with a local embedding model and requires use of the graphics card, the LANGCHAIN branch requires setting up a pinecone vector database named `main` with `1536` dimensions and `dot product` as the metric. 
