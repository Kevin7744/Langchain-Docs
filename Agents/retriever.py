# A retriver over some data of our own.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import faiss
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

loader = WebBaseLoader("https://developer.safaricom.co.ke/APIs/MpesaExpressSimulate")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = faiss.from_documents(documents, OpenAIEmbeddings)
retriever = vector.as_retriever()


retriever.get_relevant_documents("what is mpesa express")[0]

