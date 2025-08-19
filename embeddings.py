from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

persist_directory = "chroma_db"
collection_name="erdem_cafe"
embedding_tool = OpenAIEmbeddings()

vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory=persist_directory,
    embedding_function=embedding_tool
)