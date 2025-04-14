#chatbot.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import logging
import os
import threading
import numpy as np 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GENERATIVE_MODEL_NAME = "google/gemma-3-1b-it"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32 
CONTEXT_K = 3 
MAX_NEW_TOKENS = 250 



index_lock = threading.Lock()

loaded_index = None
loaded_documents = None
last_index_mtime = 0
last_metadata_mtime = 0



try:
    logger.info(f"Loading tokenizer: {GENERATIVE_MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_MODEL_NAME)
    logger.info("Tokenizer loaded.")

    logger.info(f"Loading generative model: {GENERATIVE_MODEL_NAME} on {DEVICE} with dtype {TORCH_DTYPE}...")
    model = AutoModelForCausalLM.from_pretrained(
        GENERATIVE_MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        
        
    ).to(DEVICE)
    model.eval() 
    logger.info("Generative model loaded.")

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    logger.info("Embedding model loaded.")

except Exception as e:
    logger.error(f"Error loading models: {e}", exc_info=True)
    raise SystemExit("Failed to load necessary models.")



def load_index_and_metadata():
    """
    Loads the FAISS index and metadata from disk if they exist and have changed.
    Protected by a lock for thread safety during reload checks. More robust logic.
    """
    global loaded_index, loaded_documents, last_index_mtime, last_metadata_mtime
    current_index_mtime = 0
    current_metadata_mtime = 0
    index_changed = False
    metadata_changed = False
    needs_reload = False

    
    try:
        if os.path.exists(INDEX_FILE):
            current_index_mtime = os.path.getmtime(INDEX_FILE)
            if current_index_mtime > last_index_mtime:
                logger.debug(f"Index file changed (mtime {current_index_mtime} > {last_index_mtime})")
                index_changed = True
                needs_reload = True
        elif loaded_index is not None: 
             logger.info(f"Index file {INDEX_FILE} deleted. Needs unload.")
             index_changed = True 
             needs_reload = True

        if os.path.exists(METADATA_FILE):
            current_metadata_mtime = os.path.getmtime(METADATA_FILE)
            if current_metadata_mtime > last_metadata_mtime:
                logger.debug(f"Metadata file changed (mtime {current_metadata_mtime} > {last_metadata_mtime})")
                metadata_changed = True
                needs_reload = True
        elif loaded_documents is not None: 
            logger.info(f"Metadata file {METADATA_FILE} deleted. Needs unload.")
            metadata_changed = True 
            needs_reload = True

        
        if loaded_index is None or loaded_documents is None:
             needs_reload = True

        
        if not needs_reload:
            
            return loaded_index, loaded_documents

    except OSError as e:
         logger.error(f"Error checking file status before lock: {e}", exc_info=True)
         
         needs_reload = True
         
         index_changed = True
         metadata_changed = True


    
    logger.debug(f"Acquiring index lock. Needs reload: {needs_reload}")
    with index_lock:
        logger.debug("Index lock acquired.")
        
        try:
            local_last_index_mtime = last_index_mtime
            local_last_metadata_mtime = last_metadata_mtime

            if os.path.exists(INDEX_FILE):
                current_index_mtime = os.path.getmtime(INDEX_FILE)
                
                if current_index_mtime > local_last_index_mtime or loaded_index is None:
                    index_changed = True
                else: 
                    index_changed = False
            elif loaded_index is not None: 
                index_changed = True 
            else: 
                index_changed = False

            if os.path.exists(METADATA_FILE):
                current_metadata_mtime = os.path.getmtime(METADATA_FILE)
                 
                if current_metadata_mtime > local_last_metadata_mtime or loaded_documents is None:
                     metadata_changed = True
                else: 
                    metadata_changed = False
            elif loaded_documents is not None: 
                metadata_changed = True 
            else: 
                 metadata_changed = False

        except OSError as e:
            logger.error(f"Error checking file status inside lock: {e}", exc_info=True)
            
            index_changed = True
            metadata_changed = True


        
        if index_changed:
            if os.path.exists(INDEX_FILE):
                try:
                    logger.info(f"Reloading FAISS index from {INDEX_FILE}...")
                    loaded_index = faiss.read_index(INDEX_FILE)
                    
                    last_index_mtime = current_index_mtime
                    logger.info(f"FAISS index reloaded. Total vectors: {loaded_index.ntotal if loaded_index else 'N/A'}")
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {e}", exc_info=True)
                    loaded_index = None 
                    last_index_mtime = 0 
            else:
                if loaded_index is not None: 
                    logger.warning(f"FAISS index file not found: {INDEX_FILE}. Unloading index.")
                loaded_index = None
                last_index_mtime = 0 

        if metadata_changed:
             if os.path.exists(METADATA_FILE):
                try:
                    logger.info(f"Reloading metadata from {METADATA_FILE}...")
                    with open(METADATA_FILE, "rb") as f:
                        loaded_documents = pickle.load(f)
                    
                    last_metadata_mtime = current_metadata_mtime
                    logger.info(f"Metadata reloaded. Number of documents: {len(loaded_documents) if loaded_documents else 'N/A'}")
                except (IOError, pickle.PickleError, EOFError) as e: 
                    logger.error(f"Error loading metadata file: {e}", exc_info=True)
                    loaded_documents = None 
                    last_metadata_mtime = 0 
             else:
                if loaded_documents is not None: 
                    logger.warning(f"Metadata file not found: {METADATA_FILE}. Unloading documents.")
                loaded_documents = None
                last_metadata_mtime = 0 

        logger.debug("Index lock released.")

    
    
    return loaded_index, loaded_documents


def get_context(query: str, k: int = CONTEXT_K) -> str | None:
    """Retrieves relevant context from the index for a given query."""
    logger.debug(f"Getting context for query: '{query[:50]}...'")
    index, documents = load_index_and_metadata() 

    if index is None or documents is None:
        logger.warning("Index or documents not loaded. Cannot retrieve context.")
        return None
    if index.ntotal == 0:
        logger.warning("Index is loaded but empty (ntotal=0). Cannot retrieve context.")
        return None

    if not isinstance(query, str) or not query:
        logger.warning("Received empty or invalid query for context retrieval.")
        return None

    if k <= 0:
         logger.warning(f"Requested k={k} context items, which is not positive. Returning None.")
         return None

    try:
        logger.debug(f"Encoding query with {EMBEDDING_MODEL_NAME}...")
        
        
        query_embedding = embedder.encode(
            [query],
            convert_to_tensor=False,
            convert_to_numpy=True
        ).astype(np.float32)
        logger.debug("Query encoded. Searching index...")

        
        actual_k = min(k, index.ntotal)
        if actual_k < k:
            logger.warning(f"Requested k={k} but index only has {index.ntotal} vectors. Using k={actual_k}.")
        if actual_k == 0: 
            logger.warning("Index is searchable but actual_k is 0. Returning None.")
            return None

        
        
        D, I = index.search(query_embedding, actual_k)
        logger.debug(f"Index search completed. Found indices: {I[0]}, Distances: {D[0]}")

        
        context_docs = []
        valid_indices_found = False
        
        for idx_pos, doc_index in enumerate(I[0]):
            
            if doc_index < 0:
                logger.warning(f"FAISS returned invalid index -1 at position {idx_pos}. Skipping.")
                continue
            
            if 0 <= doc_index < len(documents):
                context_docs.append(documents[doc_index])
                valid_indices_found = True
            else:
                
                logger.error(f"Invalid index {doc_index} found in search results (document count: {len(documents)}). Possible index/metadata mismatch! Skipping.")

        if not valid_indices_found:
             logger.warning(f"No valid document indices found for the search results: {I[0]}. Returning None.")
             return None

        
        
        
        context = "\n\n".join(context_docs)
        logger.debug(f"Retrieved context ({len(context_docs)} snippets):\n{context[:200]}...")
        return context

    except faiss.FaissException as e:
        logger.error(f"FAISS error during context retrieval: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during context retrieval: {e}", exc_info=True)
        return None


def generate_answer(query: str) -> str:
    """Generates an answer to the query using retrieved context and the LLM."""
    logger.info(f"Generating answer for query: '{query[:50]}...'")

    if not isinstance(query, str) or not query:
        logger.error("Received empty or invalid query for answer generation.")
        return "Please provide a valid question."

    context = get_context(query, k=CONTEXT_K)
    
    

    
    fallback_message = "I don't have enough information in the provided context to answer this question."

    if context is None:
        logger.warning("No context retrieved or context retrieval failed. Responding with fallback message.")
        
        return fallback_message
    else:
        
        
        prompt = f"""Answer the following question using *only* the provided context. Do not add any information, examples, or code not found in the context below.

Context:
{context}

Question:
{query}

If the context does not contain the information needed to answer the question, respond *only* with the sentence: "{fallback_message}"
Otherwise, provide the answer based *strictly* and *solely* on the provided context.

Answer:"""
        

    logger.debug(f"Constructed prompt (length: {len(prompt)} chars)")

    try:
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=model.config.max_position_embeddings, 
            truncation=True, 
            padding=False 
        ).to(DEVICE) 

        if inputs.input_ids.shape[-1] == model.config.max_position_embeddings:
             logger.warning("Input prompt was truncated to model's max length. Context or question might be incomplete.")

        logger.info("Generating response with LLM...")
        with torch.no_grad():
            
            output = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, 
                top_p=None,      
                top_k=None,      
                
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=tokenizer.eos_token_id 
            )
        logger.info("LLM generation complete.")

        
        
        response_ids = output[0][inputs.input_ids.shape[-1]:]
        
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        logger.info(f"Generated response: {response[:100]}...")

        
        if not response:
            logger.warning("LLM generated an empty response.")
            return fallback_message 

        return response

    except Exception as e:
        logger.error(f"Error during answer generation: {e}", exc_info=True)
        
        return "Sorry, I encountered an error while generating the answer. Please try again later."