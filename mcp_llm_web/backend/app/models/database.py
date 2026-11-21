import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, AsyncGenerator
from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Text,
    Integer,
    ForeignKey,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.future import select
from app.core.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    messages = relationship(
        "MessageModel", back_populates="session", cascade="all, delete-orphan"
    )


class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    tool_results = Column(Text, nullable=True)

    session = relationship("SessionModel", back_populates="messages")


class FileModel(Base):
    __tablename__ = "files"

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(512), nullable=False)
    uploaded_at = Column(DateTime, nullable=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


class DatabaseService:
    @staticmethod
    async def create_session(session_id: str, title: str):
        async with async_session_maker() as session:
            session_obj = SessionModel(
                id=session_id,
                title=title,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(session_obj)
            await session.commit()
            await session.refresh(session_obj)
            return session_obj

    @staticmethod
    async def get_session(session_id: str) -> Optional[SessionModel]:
        async with async_session_maker() as session:
            result = await session.execute(
                select(SessionModel).where(SessionModel.id == session_id)
            )
            return result.scalar_one_or_none()

    @staticmethod
    async def update_session_title(session_id: str, title: str):
        async with async_session_maker() as session:
            result = await session.execute(
                select(SessionModel).where(SessionModel.id == session_id)
            )
            session_obj = result.scalar_one_or_none()
            if session_obj:
                session_obj.title = title
                session_obj.updated_at = datetime.now()
                await session.commit()

    @staticmethod
    async def delete_session(session_id: str):
        async with async_session_maker() as session:
            result = await session.execute(
                select(SessionModel).where(SessionModel.id == session_id)
            )
            session_obj = result.scalar_one_or_none()
            if session_obj:
                await session.delete(session_obj)
                await session.commit()

    @staticmethod
    async def list_sessions() -> List[SessionModel]:
        async with async_session_maker() as session:
            result = await session.execute(
                select(SessionModel).order_by(SessionModel.updated_at.desc())
            )
            return list(result.scalars().all())

    @staticmethod
    async def add_message(
        session_id: str,
        role: str,
        content: str,
        tool_results: Optional[str] = None,
    ) -> MessageModel:
        async with async_session_maker() as session:
            result = await session.execute(
                select(SessionModel).where(SessionModel.id == session_id)
            )
            session_obj = result.scalar_one_or_none()

            if session_obj:
                session_obj.updated_at = datetime.now()

            message = MessageModel(
                session_id=session_id,
                role=role,
                content=content,
                timestamp=datetime.now(),
                tool_results=tool_results,
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)
            return message

    @staticmethod
    async def get_message(session_id: str, message_id: int) -> Optional[MessageModel]:
        async with async_session_maker() as session:
            result = await session.execute(
                select(MessageModel).where(
                    MessageModel.session_id == session_id, MessageModel.id == message_id,
                )
            )
            return result.scalar_one_or_none()

    @staticmethod
    async def get_messages(session_id: str) -> List[MessageModel]:
        async with async_session_maker() as session:
            result = await session.execute(
                select(MessageModel)
                .where(MessageModel.session_id == session_id)
                .order_by(MessageModel.timestamp)
            )
            return list(result.scalars().all())

    @staticmethod
    async def add_file(
        file_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        file_path: str,
    ) -> FileModel:
        async with async_session_maker() as session:
            file_obj = FileModel(
                id=file_id,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                file_path=file_path,
                uploaded_at=datetime.now(),
            )
            session.add(file_obj)
            await session.commit()
            await session.refresh(file_obj)
            return file_obj

    @staticmethod
    async def get_file(file_id: str) -> Optional[FileModel]:
        async with async_session_maker() as session:
            result = await session.execute(
                select(FileModel).where(FileModel.id == file_id)
            )
            return result.scalar_one_or_none()

    @staticmethod
    async def list_files() -> List[FileModel]:
        async with async_session_maker() as session:
            result = await session.execute(
                select(FileModel).order_by(FileModel.uploaded_at.desc())
            )
            return list(result.scalars().all())

    @staticmethod
    async def delete_file(file_id: str):
        async with async_session_maker() as session:
            result = await session.execute(
                select(FileModel).where(FileModel.id == file_id)
            )
            file_obj = result.scalar_one_or_none()
            if file_obj:
                await session.delete(file_obj)
                await session.commit()


db_service = DatabaseService()
