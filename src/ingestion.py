from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

print(docs_list)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# Initialize Sentence Transformer embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create and persist the vector store
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
    persist_directory="./.chroma",
)

# Create retriever
retriever = vectorstore.as_retriever()

if __name__ == "__main__":
    # Test retrieval
    query = "What are AI agents?"
    results = retriever.invoke(query)

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content)
