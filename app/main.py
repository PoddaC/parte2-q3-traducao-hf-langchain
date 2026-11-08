from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import translation
from app.services.translation_service import get_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_chain()  # startup: carrega o modelo uma única vez
    yield


app = FastAPI(title="Tradução EN→DE com LangChain + HuggingFace", version="1.0.0", lifespan=lifespan)
app.include_router(translation.router)
