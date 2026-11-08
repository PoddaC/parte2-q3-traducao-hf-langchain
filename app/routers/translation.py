from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from app.models.translation import TranslateRequest, TranslateResponse
from app.services.translation_service import MODEL_ID, get_chain

router = APIRouter(tags=["tradução"])


@router.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest, chain=Depends(get_chain)) -> TranslateResponse:
    # Inferência local é CPU/GPU-bound e bloqueante: executa em threadpool para não travar o event loop.
    translation = await run_in_threadpool(chain.invoke, {"text": req.text})
    return TranslateResponse(original=req.text, translation=translation.strip(), model=MODEL_ID)
