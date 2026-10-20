from typing import List, TypedDict
from xml.dom.minidom import Document


class GraphState(TypedDict):
    """
    Represents a graph state in a graph
    question : question
    generation : LLM Generation
    web_search : wheather to search
    documents : list of documents
    """
    question: str
    generation: int
    web_search: bool
    documents: List[Document]
