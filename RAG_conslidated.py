import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChartPromtTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)

document = TextLoader("product-data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_splitter.split_documents(document)
vector_store=Chroma.from_documents(chunks,embeddings)
retriever = vector_store.as.retrieval()

prompt_template = ChatPromptTemplate.from_message (
  [
    ("system","""You are assistance for answering questions.
                  Use the provided context to respond. if the answer
                  is not clear,aknowledge that you dont know.
                  Limit your response to three concise sentences.
                  {context}

                  """),
                  ("human","{input}")
  ]
)

qa_chain = create_stuff_documents_chain(llm,prompt_template)
reg_chain = input("Your Questions")

Print("Chat with Document")
question = input("Your Questions")

if question:
  response = reg_chain.invoke("input":question)
  print(response['answer']
        

                 
                  


     


db = Chroma.from_documents(chunks, llm)

text = input("Enter the query")
embedding_vector = llm.embed_query(text)
docs = db.similarity_search_by_vector(embedding_vector)

for doc in docs:
  print(doc.page_content)
