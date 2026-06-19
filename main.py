import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel
from translator import translate_text, AVAILABLE_MODELS, DEFAULT_MODEL

app = FastAPI(
    title="Text Orbit",
    description="Translate text into 10 languages using AI",
    version="1.0.0",
)


# --- Models ---

class TranslateRequest(BaseModel):
    text: str
    tone: str = "formal"


class TranslationResponse(BaseModel):
    Tamil: str
    English: str
    Kannada: str
    Malayalam: str
    Arabic: str
    Telugu: str
    Thanglish: str
    Hindi: str
    French: str
    German: str


# --- Routes (must be registered BEFORE static mount) ---

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.post(
    "/translate",
    response_model=TranslationResponse,
    summary="Translate text into 10 languages",
    description=(
        "Translate the given text into Tamil, English, Kannada, Malayalam, "
        "Arabic, Telugu, Thanglish, Hindi, French, and German. "
        "Tone can be 'formal' or 'casual'."
    ),
    operation_id="translate_text",
)
async def translate(request: TranslateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        result = await translate_text(request.text, request.tone)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.get("/models", summary="List available models", operation_id="list_models")
async def list_models():
    return {"models": AVAILABLE_MODELS, "default": DEFAULT_MODEL}


@app.get("/health", summary="Health check", operation_id="health_check")
async def health():
    return {"status": "ok"}


# --- MCP (Streamable HTTP transport) ---
mcp = FastApiMCP(app)
mcp.mount_http()

# --- Static files (mount LAST) ---
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Run directly ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
