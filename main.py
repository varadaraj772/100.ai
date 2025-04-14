#main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field
import logging
import asyncio
import os


from fastapi.middleware.cors import CORSMiddleware


from chatbot import generate_answer, INDEX_FILE as CHATBOT_INDEX_FILE
from db import init_db, add_knowledge
from embed_and_index import build_index


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="RAG Chatbot API",
    description="API for interacting with the RAG chatbot and managing its knowledge base.",
    version="1.0.0",
)


origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "null",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str = Field(
        ..., min_length=3, description="The question to ask the chatbot."
    )


class DataInput(BaseModel):
    text: str = Field(
        ..., min_length=10, description="The piece of knowledge text to add."
    )


class ChatResponse(BaseModel):
    answer: str


class StatusResponse(BaseModel):
    status: str
    detail: str | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize database connection and schema on startup."""
    logger.info("Application startup...")
    try:
        await init_db()

        if not os.path.exists(CHATBOT_INDEX_FILE):
            logger.warning(
                f"Index file '{CHATBOT_INDEX_FILE}' not found during startup. Running initial index build..."
            )
            await build_index()
            logger.info("Initial index build complete.")
        else:
            logger.info(
                f"Index file '{CHATBOT_INDEX_FILE}' found. Skipping initial build during startup."
            )

    except Exception as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        raise RuntimeError(
            "Failed to initialize database or index during startup."
        ) from e
    logger.info("Application startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown (if necessary)."""
    logger.info("Application shutdown...")

    logger.info("Application shutdown complete.")


@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    """Receives a question and returns the chatbot's answer."""
    logger.info(f"Received chat request: '{query.question[:50]}...'")
    try:

        answer = generate_answer(query.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        logger.error(f"Unhandled error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while processing the chat request.",
        )


@app.post("/add-data", response_model=StatusResponse)
async def add_data_endpoint(data: DataInput, background_tasks: BackgroundTasks):
    """Adds new knowledge text to the database and triggers background re-indexing."""
    logger.info(f"Received request to add data: '{data.text[:50]}...'")

    success = await add_knowledge(data.text)
    if not success:
        logger.error("Failed to add data to the database.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add data to the database. Check logs for details.",
        )

    logger.info("Scheduling background task to rebuild FAISS index and metadata...")
    background_tasks.add_task(build_index)
    logger.info("Background index rebuild task scheduled.")

    return StatusResponse(
        status="success",
        detail="Data added successfully. Index rebuild initiated in the background. Changes will be reflected shortly after the task completes.",
    )


@app.get("/health", response_model=StatusResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint. Indicates the API is running."""
    return StatusResponse(status="ok", detail="Service is running.")
