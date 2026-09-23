"""A: public catalog app shell. No merchant runtime or private-data routes."""
from fastapi import FastAPI

app = FastAPI(title="Central catalog scaffold", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/healthz")
def healthz():
    return {"service": "central", "status": "scaffold", "ready": False}
