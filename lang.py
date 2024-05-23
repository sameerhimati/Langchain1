import os
import openai
import sys
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv, find_dotenv

from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma

import pandas as pd
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch

client = MongoClient('mongodb://localhost:27017')
db = client['langchain']
collection = db['resumes']
vector_search_index = "vector_index"

embedding = OpenAIEmbeddings()

sys.path.append('../..')


_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]


loader = PyPDFLoader("Profile.pdf")
pages = loader.load()

page = pages[2]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20
)
splits = text_splitter.split_documents(pages)
len(splits)


vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents = splits,
    embedding = OpenAIEmbeddings(disallowed_special=()),
    collection = collection,
    index_name = vector_search_index
)

retriever = vector_search.as_retriever(
   search_type = "similarity",
   search_kwargs = {"k": 10, "score_threshold": 0.75}
)

# Define a prompt template
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")


def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)
# Construct a chain to answer questions on your data
rag_chain = (
   { "context": retriever | format_docs, "question": RunnablePassthrough()}
   | custom_rag_prompt
   | llm
   | StrOutputParser()
)

#  Prompt the chain
question = "Where did I work in 2022?"
answer = rag_chain.invoke(question)
print("Question: " + question)
print("Answer: " + answer)
# Return source documents
documents = retriever.get_relevant_documents(question)
print("\nSource documents:")
print(documents)