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
