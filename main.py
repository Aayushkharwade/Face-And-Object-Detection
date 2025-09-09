from fastapi import FastAPI

app = FastAPI(title="Date-Time Based RAG Chatbot Pipeline", version="2.0.0")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)