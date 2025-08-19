from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import Document
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from embeddings import vectorstore
load_dotenv()

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

model = ChatOpenAI(model="gpt-4.1-nano", max_tokens=100, temperature=0.3)


class ChatbotState(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list[Document]
    in_scope: bool

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

def topic_decider(state: ChatbotState):
    user_request = state["messages"][-1].content
    results = vectorstore.similarity_search_with_score(user_request, k=1)
    if not results:
        return {"in_scope": False}
    else:
        score = results[0][1]
        print(score)
        in_scope = score <= 0.45
        return {"in_scope":in_scope}

def router_by_scope(state: ChatbotState) -> Literal["in","out"]:
    in_scope = state["in_scope"]
    if in_scope:
        return "in"
    return "out"

def responder(state: ChatbotState):
    return {"messages": [AIMessage(content="I'm sorry, I can't answer this question.")]}

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
TOPIC_DECIDER = "topic-decider"
RESPONDER = "responder"

graph = StateGraph(ChatbotState)

graph.add_node(TOPIC_DECIDER, topic_decider)
graph.add_node(CHATBOT, chatbot)
graph.add_node(RETRIEVER, retriever)
graph.add_node(RESPONDER, responder)

graph.set_entry_point(TOPIC_DECIDER)

graph.add_edge(RETRIEVER, CHATBOT)

graph.add_conditional_edges(TOPIC_DECIDER, router_by_scope,{"in": RETRIEVER, "out":RESPONDER})

app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    else:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        print(result["messages"][-1].content)
