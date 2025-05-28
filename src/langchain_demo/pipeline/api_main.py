from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form

import logging
import uvicorn


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

@app.get("/")
async def home() -> Dict[str, str]:
    """
    Root endpoint that verifies the application is running.
    
    Returns:
        Dict[str, str]: A dictionary containing a status message
    """
    return {"message": "App is running"}


@app.post("/full-extract/")
async def full_extract(
    file: UploadFile = File(None),
    info_to_extract: List[str] = Form(...),  
    system_prompt: str = Form(...),  
):
    raise NotImplementedError("This endpoint is not implemented yet.")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
