from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World", 'lulu': 0}
