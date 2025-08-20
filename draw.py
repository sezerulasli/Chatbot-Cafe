from langchain_core.runnables.graph import MermaidDrawMethod
from app import app

png = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
with open("graph.png", "wb") as f:
    f.write(png)
print("Saved graph.png")