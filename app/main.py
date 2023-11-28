import uvicorn 
import pandas as pd
import torch
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from app.location_predictor import LocationPredictor
from typing import List


# model_checkpoint_uk = "app/models/model-uk/checkpoint-14096/"
# model_checkpoint_ru = "app/models/model-ru/checkpoint-12500/"
model_checkpoint_uk = "app/models/models_onnx/model_uk_onnx_optimized/"
model_checkpoint_ru = "app/models/models_onnx/model_ru_onnx/"
stoprows = "app/models/models_onnx/stoprows.json"


class InputTexts(BaseModel):
    texts: List[str]


models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start-up: Load saved models
    models["locations-extractor"] = LocationPredictor(
        chekpoint_path_uk=model_checkpoint_uk,
        chekpoint_path_ru=model_checkpoint_ru,
        stoprows_path=stoprows,
        onnx=True
    )
    yield
    # Shutdown: free memory 
    models.clear()
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