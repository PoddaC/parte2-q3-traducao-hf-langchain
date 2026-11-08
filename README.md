# Parte 2 – Questão 3: Tradução EN → DE com LangChain + HuggingFace (FastAPI)

API que recebe texto em inglês e devolve a tradução em alemão. O LangChain gerencia a chamada ao modelo
[`Helsinki-NLP/opus-mt-en-de`](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) por meio de `HuggingFacePipeline` (execução **local**).

## Arquitetura
```mermaid
flowchart LR
    C[Cliente] -->|"POST /translate {text}"| A[FastAPI<br/>async + Pydantic]
    A -->|threadpool| CH
    subgraph LC [LangChain - LCEL]
      CH[PromptTemplate<br/>"{text}"] --> H[HuggingFacePipeline<br/>task=translation] --> PA[StrOutputParser]
    end
    H --> T[transformers pipeline<br/>MarianMT opus-mt-en-de]
    T <-->|download 1ª vez| HUB[(HuggingFace Hub)]
    PA -->|texto| A
    A -->|"JSON {original, translation, model}"| C
```

## Estrutura
```
app/
  main.py                           # app + lifespan (carrega o modelo no startup) + routers
  routers/translation.py            # POST /translate (threadpool)
  services/translation_service.py   # prompt + HuggingFacePipeline (CPU/GPU) + parser
  models/translation.py             # schemas Pydantic
tests/test_api.py
```

## Como executar
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
curl -X POST http://127.0.0.1:8000/translate -H "Content-Type: application/json" -d '{"text": "Good morning, how are you?"}'
```
O modelo (~300 MB) é baixado na primeira execução e fica em cache (`~/.cache/huggingface`). Docs: <http://127.0.0.1:8000/docs>.

## Testes
```bash
python -m pytest -q
```
Usam um LLM falso no lugar do modelo real (sem download).
