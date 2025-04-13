import os
from sqlmodel import SQLModel, create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import logging

from models import Knowledge


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")


engine = create_async_engine(DATABASE_URL, echo=False, future=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Initializes the database schema."""
    logger.info("Initializing database schema...")
    try:
        async with engine.begin() as conn:

            await conn.run_sync(SQLModel.metadata.create_all)
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database schema: {e}")
        raise


async def get_knowledge() -> list[str]:
    """Retrieves all knowledge texts from the database."""
    logger.info("Fetching knowledge from database...")
    documents = []
    try:
        async with async_session() as session:
            result = await session.execute(select(Knowledge))
            documents = [row.text for row in result.scalars().all()]
        logger.info(f"Fetched {len(documents)} knowledge documents.")

        return documents
    except SQLAlchemyError as e:
        logger.error(f"Database error fetching knowledge: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching knowledge: {e}")
        return []


async def add_knowledge(text: str) -> bool:
    """Adds a new knowledge text to the database."""
    logger.info("Adding new knowledge to database...")
    try:
        async with async_session() as session:
            async with session.begin():
                knowledge_entry = Knowledge(text=text)
                session.add(knowledge_entry)

        logger.info("Successfully added new knowledge.")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database error adding knowledge: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error adding knowledge: {e}")
        return False
