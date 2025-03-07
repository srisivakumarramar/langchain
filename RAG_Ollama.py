# product-data.txt,raq_questions.txt are used for this code
import os
from import langchain_Ollama,OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate,MessagePlaceHolder
from langchain.chains import create_retrieval_chain,create_histroy_aware_retriever
from langchain.chains.combine_documents import create_stuff
import streamlit as st
from langchain_community.chart_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = OllamaEmbeddings(model="llama3.2")
embeddings=OllamaEmbeddings(model="llama3.2")

document = TextLoader("Legal_Document_Analysis_Data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever

prompt_template = ChatPromptTemplate.from_message (
  [
    ("system","""You are assistance for answering questions.
                  Use the provided context to respond. if the answer
                  is not clear,aknowledge that you dont know.
                  Limit your response to three concise sentences.
                  {context}

                  """),
                  MessagePlaceHolder(variable_name="Chat_history"),
                  ("human","{input}")
  ]
)

history_aware_retriever = create_history_aware_retriever(11m,retriever,prompt_template)
qa_chain = create_stuff_documents_chain(llm,prompt_template)
rag_chain = create_retrieval_chain(history_aware_retrieva,qa_chain)

history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
  rag_chain,
  lambda session-id:history_for_chain,
  input_message_key="input",
  history_message_key="chat_history"
)

st.write("Chat with Document")
question =st.text_input("Your Questions")

if question:
  response = chain_with_history.invoke({"input":question},{"configurable":{"session_id":"abc123"}})
  st.write(response['answer']
