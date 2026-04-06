from fastapi import FastAPI

app = FastAPI(title="Salary Prediction API", version="0.1.0")


@app.get("/")
def root():
    return {"message": "Salary Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}