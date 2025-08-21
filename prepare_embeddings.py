from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
import os, shutil

load_dotenv()

Menu = [
  {"name": "Espresso", "category": "coffee", "notes": "concentrated"},
  {"name": "Cappuccino", "category": "coffee", "notes": "espresso, hot milk, and steamed milk foam"},
  {"name": "Americano", "category": "coffee", "notes": "espresso and hot water"},
  {"name": "Cafe Latte", "category": "coffee", "notes": "espresso and steamed milk"},
  {"name": "Turkish Coffee", "category": "coffee", "notes": "prepared with finely ground coffee beans"},
  {"name": "Macchiato", "category": "coffee", "notes": "espresso with a small amount of milk foam"},
  {"name": "Mocha", "category": "coffee", "notes": "chocolate-flavored latte"},
  {"name": "Filter Coffee", "category": "coffee", "notes": "brewed by passing hot water through ground coffee beans"},
  {"name": "Black Tea", "category": "tea", "notes": "fermented tea leaves"},
  {"name": "Green Tea", "category": "tea", "notes": "unfermented tea leaves"},
  {"name": "White Tea", "category": "tea", "notes": "made from young tea leaves and buds"},
  {"name": "Oolong Tea", "category": "tea", "notes": "partially fermented tea leaves"},
  {"name": "Rooibos Tea", "category": "tea", "notes": "caffeine-free herbal tea made from the red bush plant"},
  {"name": "Orange Juice", "category": "juice", "notes": "from freshly squeezed oranges"},
  {"name": "Apple Juice", "category": "juice", "notes": "from freshly squeezed apples"},
  {"name": "Pomegranate Juice", "category": "juice", "notes": "from freshly squeezed pomegranates"},
  {"name": "Sour Cherry Juice", "category": "juice", "notes": "from freshly squeezed sour cherries"},
  {"name": "Pineapple Juice", "category": "juice", "notes": "from freshly squeezed pineapples"},
  {"name": "Cola", "category": "soda", "notes": "caramel-colored, caffeinated carbonated beverage"},
  {"name": "Lemon-Lime Soda", "category": "soda", "notes": "lemon-flavored carbonated beverage"},
  {"name": "Tonic Water", "category": "soda", "notes": "a bitter carbonated drink containing quinine"},
  {"name": "Margarita", "category": "cocktail", "notes": "tequila, orange liqueur, and lime juice"},
  {"name": "Martini", "category": "cocktail", "notes": "gin and vermouth"},
  {"name": "Mojito", "category": "cocktail", "notes": "rum, sugar, lime juice, soda water, and mint"},
  {"name": "Pina Colada", "category": "cocktail", "notes": "rum, coconut cream, and pineapple juice"},
  {"name": "Sex On The Beach", "category": "cocktail", "notes": "vodka, peach schnapps, orange juice, and cranberry juice"},
  {"name": "Lemonade", "category": "other", "notes": "lemon juice, water, and sugar"},
  {"name": "Iced Tea", "category": "other", "notes": "chilled tea"},
  {"name": "Milkshake", "category": "other", "notes": "a sweet, cold beverage made with milk, ice cream, and syrups"},
  {"name": "Water", "category": "other", "notes": "pure and refreshing"}   
]

Drinks = [
    Document(
        page_content=f"{m['name']} - {m['category']} - {m['notes']}",
        metadata=m
    )
    for m in Menu
]

persist_directory = "chroma_db"

if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

collection_name="erdem_cafe"

embedding_tool = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(Drinks, embedding_tool, persist_directory=persist_directory, collection_name=collection_name)

print("RAG seti oluşturuldu.")