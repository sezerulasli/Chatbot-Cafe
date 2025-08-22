from chatbot.app import app as chatbot
from langchain_core.messages import HumanMessage
from starlette.concurrency import run_in_threadpool

async def get_chatbot_reponse(user_id: str, user_message: str) -> str:   ## LLM çağrısını yapacak fonksiyon.
    config = {"configurable": {"thread_id": user_id}}
    result = await run_in_threadpool( ## SqliteSaver senkron çalıştığı için bu history sistemini API'da asenkron çağırmak için kullanılan bir metot.
        chatbot.invoke,
        {"messages": [HumanMessage(content=user_message)]} ,
        config=config)
    return result["messages"][-1].content