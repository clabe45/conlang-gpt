from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceBgeEmbeddings
from langchain.storage import LocalFileStore


def get_embeddings_model(model_name="BAAI/bge-small-en"):
    fs = LocalFileStore(".conlang/cache/embeddings")
    hf_embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)
    return CacheBackedEmbeddings.from_bytes_store(
        hf_embeddings, fs, namespace=hf_embeddings.model_name
    )
