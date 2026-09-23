"""C: coordinator app shell. Submission routes stay absent until implemented."""
from fastapi import FastAPI

app = FastAPI(title="FL coordinator scaffold", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/healthz")
def healthz():
    return {"service": "coordinator", "status": "scaffold", "ready": False}
