#embed_and_index.py
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import asyncio
import logging
import os


from db import get_knowledge, init_db


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"


try:
    logger.info(f"Loading embedding model: {MODEL_NAME}...")
    embedder = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model loaded.")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")

    raise SystemExit("Embedding model loading failed.")


async def build_index():
    """Fetches data, creates embeddings, builds FAISS index, and saves index/metadata."""
    logger.info("Starting index building process...")

    docs = await get_knowledge()
    if not docs:
        logger.warning("No documents found in the database. Index will be empty.")

        index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())

    else:

        logger.info(f"Creating embeddings for {len(docs)} documents...")
        try:
            embeddings = embedder.encode(docs, show_progress_bar=True)
            logger.info("Embeddings created successfully.")
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return

        dim = embeddings.shape[1]
        logger.info(f"Building FAISS index with dimension {dim}...")

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        logger.info(f"FAISS index built. Total vectors: {index.ntotal}")

    try:
        logger.info(f"Saving FAISS index to {INDEX_FILE}...")
        faiss.write_index(index, INDEX_FILE)
        logger.info("FAISS index saved.")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        return

    try:
        logger.info(f"Saving metadata to {METADATA_FILE}...")
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(docs, f)
        logger.info("Metadata saved.")
    except IOError as e:
        logger.error(f"Error saving metadata file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving metadata: {e}")

    logger.info("Index building process finished.")


async def main():
    """Main async function to initialize DB and build index."""

    await init_db()
    await build_index()


if __name__ == "__main__":

    logger.info("Running embed_and_index.py script...")
    asyncio.run(main())
    logger.info("Script finished.")
