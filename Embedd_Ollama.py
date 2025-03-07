import os
from langchain_Ollama import OllamaEmbeddings

llm = OllamaEmbeddings(model="llama3.2")

text = input("Enter the text")
response = llm.embed_query(text)
print(response)
