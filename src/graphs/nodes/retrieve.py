from typing import Dict ,Any

from graphs.state import GraphState
from ingestion import retriever

def retrieve(state : GraphState) -> Dict[str, Any]:
    print("--- retrieving ---")
    question = state["question"]
    documents = retriever.retrieve(question)
    return {'question': question, 'documents': documents}

