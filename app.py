from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.schema import Document
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Chroma
load_dotenv()

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

model = ChatOpenAI(model="gpt-4.1-nano", max_tokens=100, temperature=0.3)


class ChatbotState(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list[Document]


Drinks = [
    Document(page_content="We are Erdem's Cafe. We have a menu of drinks included: Espresso, Cappuccino, Latte, Americano, Mocha, Macchiato, Hot Chocolate, Green Tea, Black Tea, Herbal Tea, Iced Coffee, Iced Latte, Lemonade, Orange Juice, Apple Juice, Mineral Water, Sparkling Water, Cola, Ginger, Tuborg Gold")
    ]

embedding_tool = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(Drinks, embedding_tool)

retriever_tool = vectorstore.as_retriever(search_type="mmr", search_kwargs = {"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system","You are expert barista And suggest drinks that there only are in our menu: {documents} according to user's request"),
    MessagesPlaceholder("history"),
    ("human","{user_request}"),
    
])
rag_chain = prompt | model

def retriever(state: ChatbotState):
    user_request = state["messages"][-1].content
    documents = retriever_tool.invoke(user_request)
    
    return {"documents": documents}


def chatbot(state: ChatbotState):
    user_request = state["messages"][-1].content
    documents = state["documents"]
    history = state["messages"]

    response = rag_chain.invoke({
        "documents": documents,
        "user_request": user_request,
        "history": history
    })

    return {"messages": [response]}

CHATBOT = "chatbot"
RETRIEVER = "retriever"

graph = StateGraph(ChatbotState)

graph.add_node(CHATBOT, chatbot)
graph.add_node(RETRIEVER, retriever)

graph.set_entry_point(RETRIEVER)

graph.add_edge(START, RETRIEVER)
graph.add_edge(RETRIEVER, CHATBOT)


app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    else:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        print(result["messages"][-1].content)
