import os, shutil, tempfile, requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from app.pipeline import load_singletons, detect_card_names, detect_card_ids

app = FastAPI(title="Card Detection API", version="1.0.0")

# CORS (so Express/Frontends can call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class DetectResponse(BaseModel):
    card_names: list[str]

class DetectIDsResponse(BaseModel):
    card_ids: list[str]

class UrlIn(BaseModel):
    image_url: HttpUrl

@app.on_event("startup")
def _startup():
    # Warm models & Pinecone once
    load_singletons()

@app.get("/health")
def health():
    return {"status": "ok"}

# ----- upload file -> names
@app.post("/detect/names", response_model=DetectResponse)
async def detect_names(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        names = detect_card_names(tmp_path)
        return {"card_names": names}
    finally:
        try: os.remove(tmp_path)
        except: pass

# ----- upload file -> ids
@app.post("/detect/ids", response_model=DetectIDsResponse)
async def detect_ids(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        ids = detect_card_ids(tmp_path)
        return {"card_ids": ids}
    finally:
        try: os.remove(tmp_path)
        except: pass

# ----- URL input as alternative
@app.post("/detect/names-by-url", response_model=DetectResponse)
async def detect_names_by_url(body: UrlIn):
    resp = requests.get(str(body.image_url), timeout=20)
    if resp.status_code != 200:
        raise HTTPException(400, "Could not download image_url")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(resp.content)
        path = tmp.name
    try:
        names = detect_card_names(path)
        return {"card_names": names}
    finally:
        try: os.remove(path)
        except: pass
