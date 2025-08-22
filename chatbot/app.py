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
from chatbot.embeddings import vectorstore
import uuid
load_dotenv()

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

model = ChatOpenAI(model="gpt-4.1-nano", max_tokens=100, temperature=0.2)

class ChatbotState(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list[Document]
    in_scope: bool

retriever_tool = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs = {"k": 20, "score_threshold":0.3})

prompt = ChatPromptTemplate.from_messages([
    ("system","You are expert barista And suggest drinks that there only are in our (We are Erdem's Cafe) menu: {documents} according to user's request"
    "Keep your answers not so long and your suggestions must fit ONLY with products with its specifications in our menu."),
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
    threshold = 0.47
    delta = 0.04
    decider_model = ChatOpenAI(model="gpt-4.1-nano", max_tokens=10, temperature=0.2)
    score_prompt = ChatPromptTemplate.from_messages([
        ("system", "Decide ONLY if the user's request is about the cafe's drink menu or details of products. "
	    "Even if the requested items may NOT be on the menu, it is still IN-SCOPE. "
	    "Examples IN-SCOPE: 'cafe's drink menu, product details, questions about drinks related on menu. "
	    "Examples OUT-OF-SCOPE: history, politics, coding, personal info, general knowledge. "
	    "Answer strictly 'Yes' for in-scope, 'No' for out-of-scope."),
        ("human", "{user_request}")
    ])

    decider_chain = score_prompt | decider_model
    
    if not results:
        return {"in_scope": False}
    else:
        score = results[0][1]
        
        print(score)
        if score <= threshold:
            return {"in_scope":True}
        elif threshold < score < threshold + delta:
            decider_result = decider_chain.invoke({"user_request":user_request})
            final_dec = decider_result.content.strip().lower()
            print(final_dec) #- Decider'ın kontrolü.
            if final_dec == "no":
                return {"in_scope": False}
            return {"in_scope": True} 
        else:
            return {"in_scope": False}
        


def router_by_scope(state: ChatbotState) -> Literal["in","out"]: ## Sadece in ve out çıktısını vermesi için kullanılıyor.
    in_scope = state["in_scope"]
    if in_scope:
        return "in"
    return "out"

def responder(state: ChatbotState):
    return {"messages": [AIMessage(content="I'm sorry, I can't answer this question.")]}

def format_docs(docs: list[Document]) -> str:
    return "\n".join(f"-{d.page_content}"for d in docs)


def chatbot(state: ChatbotState):
    user_request = state["messages"][-1].content
    documents = state["documents"]
    history = state["messages"]
    short_history = history[-4:] # her seferinde sadece son 4 mesajı alarak LLM'e göndereceğiz.

    docs_text = format_docs(documents)
    response = rag_chain.invoke({
        "documents": docs_text,
        "user_request": user_request,
        "history": short_history
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

if __name__ == "__main__":   ## Burası chatbot'un API dışında CLI'da çalışmasını sağlayacak yer.
    thread_id = str(uuid.uuid4()) ## her oturmda yeni bir thread_id generate edileceği için history her oturumda sıfırlanacak.
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        else:
            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            print("Cafe Assistant: ", result["messages"][-1].content)
