from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document

load_dotenv()

Drinks = [
    Document(page_content="We are Erdem's Cafe. We have a menu of drinks included: Espresso, Cappuccino, Latte, Americano, Mocha, Macchiato, Hot Chocolate, Green Tea, Black Tea, Herbal Tea, Iced Coffee, Iced Latte, Lemonade, Orange Juice, Apple Juice, Mineral Water, Sparkling Water, Cola, Ginger, Tuborg Gold")
    ]

persist_directory = "chroma_db"
collection_name="erdem_cafe"
embedding_tool = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(Drinks, embedding_tool, persist_directory=persist_directory, collection_name=collection_name)
vectorstore.persist()
print("RAG seti oluşturuldu.")