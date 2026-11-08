"""LangChain + HuggingFacePipeline (modelo local) para tradução EN->DE."""
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

MODEL_ID = "Helsinki-NLP/opus-mt-en-de"

# Modelos Marian traduzem o texto cru: o prompt é apenas uma passagem ("{text}"), sem instruções.
PROMPT = PromptTemplate.from_template("{text}")

def get_device() -> int:
    """0 = primeira GPU (se houver CUDA); -1 = CPU."""
    import torch

    return 0 if torch.cuda.is_available() else -1


def build_llm():
    from langchain_huggingface import HuggingFacePipeline

    return HuggingFacePipeline.from_model_id(
        model_id=MODEL_ID,
        task="translation",
        device=get_device(),
        pipeline_kwargs={"max_length": 512},
    )


def build_chain(llm=None):
    return PROMPT | (llm or build_llm()) | StrOutputParser()


@lru_cache(maxsize=1)
def get_chain():
    return build_chain()  # carrega o modelo uma única vez
