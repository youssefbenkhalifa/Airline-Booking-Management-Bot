from fastapi import FastAPI
from chat import get_response

app = FastAPI()

@app.get("/{question}")
async def read_item(question:str):
       response, tag = get_response(question)
       return {"question":question, "response": response , "tag":tag}
