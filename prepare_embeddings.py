from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
import os, shutil

load_dotenv()

loader = CSVLoader(file_path="./documents/data.csv")
rag_data = loader.load()

Drinks = rag_data
persist_directory = "chroma_db"

if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

collection_name="erdem_cafe"

embedding_tool = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(Drinks, embedding_tool, persist_directory=persist_directory, collection_name=collection_name)

print("RAG seti oluşturuldu.")