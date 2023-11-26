import uvicorn 
import pandas as pd
import torch
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from location_predictor import LocationPredictor
from typing import List


model_checkpoint_uk = "models/model-uk/checkpoint-14096/"
model_checkpoint_ru = "models/model-ru/checkpoint-12500/"
stoprows = "models/stoprows.json"


class InputTexts(BaseModel):
    texts: List[str]


models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start-up: Load saved models
    models["locations-extractor"] = LocationPredictor(
        chekpoint_path_uk=model_checkpoint_uk,
        chekpoint_path_ru=model_checkpoint_ru,
        stoprows_path=stoprows
    )
    yield
    # Shutdown: free memory 
    del model
    torch.cuda.empty_cache()


app = FastAPI(
    title="NLP UA Locations extraction",
    description="",
    summary="Endpoint for locations extraction from the news and Telegram posts.",
    version="0.0.1",
    contact={
        "name": "Yevhenii Azarov",
        "email": "azarov.evg.ua@gmail.com",
    },
    lifespan=lifespan
)

@app.get("/")
def docs_redirect():
    return RedirectResponse("/docs")

@app.post("/api/extract_locations/")
async def extract_locations(input_texts: InputTexts):
    texts = input_texts.texts
    locations = models['locations-extractor'].predict(texts)
    return locations 


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)