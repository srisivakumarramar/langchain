# Job Listing using FAISS (facebook ai similarity serach) Vector to chuck the charcter from a document and to find matching job skill from input of the user with chroma 
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 


llm = OllamaEmbeddings(model="llama3.2")

document = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_splitter.split_documents(document)
db = FAISS.from_documents(chunks, llm)
docs = retriever.invoke(text)
text = input("Enter the query")
for doc in docs:
  print(doc.page_content)

# another method  
# retriever = db.as_retriever()
# text = input("Enter the Query")
# for doc in docs:
#    print(doc.page_content)
