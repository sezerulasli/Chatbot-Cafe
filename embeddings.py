from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

load_dotenv()

Drinks = [
    Document(page_content="We are Erdem's Cafe. We have a menu of drinks included: Espresso, Cappuccino, Latte, Americano, Mocha, Macchiato, Hot Chocolate, Green Tea, Black Tea, Herbal Tea, Iced Coffee, Iced Latte, Lemonade, Orange Juice, Apple Juice, Mineral Water, Sparkling Water, Cola, Ginger, Tuborg Gold")
    ]

embedding_tool = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(Drinks, embedding_tool)