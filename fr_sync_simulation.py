import asyncio
import logging
import random
from datetime import datetime
from typing import Dict, List

import asyncpg
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, String, create_engine, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())

    embedding = relationship(
        "Embedding", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    embedding = Column(Vector(512))
    updated_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())

    user = relationship("User", back_populates="embedding")


class FROfflineSyncService:
    def __init__(
        self, fr_online_db_url: str, fr_offline_db_url: str, sync_interval: int = 60
    ):
        self.fr_online_db_url = fr_online_db_url
        self.fr_offline_db_url = fr_offline_db_url
        self.sync_interval = sync_interval
        self.last_sync_timestamp = datetime.min

        self.online_engine = create_engine(fr_online_db_url)
        self.offline_engine = create_engine(fr_offline_db_url)

        # Create vector extension if it doesn't exist
        self.create_vector_extension(self.online_engine)
        self.create_vector_extension(self.offline_engine)

        Base.metadata.create_all(self.online_engine)
        Base.metadata.create_all(self.offline_engine)

        self.online_session = sessionmaker(bind=self.online_engine)()
        self.offline_session = sessionmaker(bind=self.offline_engine)()

    def create_vector_extension(self, engine):
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        logger.info(f"Ensured vector extension exists for {engine.url}")

    async def start(self):
        await asyncio.gather(self.run_periodic_sync(), self.listen_for_changes())

    async def run_periodic_sync(self):
        while True:
            await self.sync_databases()
            await asyncio.sleep(self.sync_interval)

    async def listen_for_changes(self):
        conn = await asyncpg.connect(self.fr_online_db_url)

        await conn.execute(
            """
            CREATE OR REPLACE FUNCTION notify_change() RETURNS TRIGGER AS $$
            DECLARE
                payload JSON;
            BEGIN
                payload = json_build_object(
                    'table', TG_TABLE_NAME,
                    'action', TG_OP,
                    'data', row_to_json(NEW)
                );
                PERFORM pg_notify('change_event', payload::text);
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            DROP TRIGGER IF EXISTS users_change_trigger ON users;
            CREATE TRIGGER users_change_trigger
            AFTER INSERT OR UPDATE OR DELETE ON users
            FOR EACH ROW EXECUTE FUNCTION notify_change();

            DROP TRIGGER IF EXISTS embeddings_change_trigger ON embeddings;
            CREATE TRIGGER embeddings_change_trigger
            AFTER INSERT OR UPDATE OR DELETE ON embeddings
            FOR EACH ROW EXECUTE FUNCTION notify_change();
        """
        )

        await conn.add_listener("change_event", self.handle_change_notification)

        logger.info("Listening for changes on the online database")
        while True:
            await asyncio.sleep(3600)

    async def handle_change_notification(self, connection, pid, channel, payload):
        logger.info(f"Change detected: {payload}")
        await self.sync_databases()

    async def sync_databases(self):
        try:
            logger.info("Starting database synchronization")
            changes = self.get_changes_since_last_sync()
            if changes:
                logger.info(f"Found {len(changes)} changes to synchronize")
                await self.apply_changes_to_offline(changes)
                logger.info(f"Applied {len(changes)} changes to the offline database")
            else:
                logger.info("No changes found to synchronize")
            self.last_sync_timestamp = datetime.now()
            logger.info(f"Sync completed at {self.last_sync_timestamp}")
        except Exception as e:
            logger.error(f"Error during synchronization: {str(e)}")

    def get_changes_since_last_sync(self) -> List[Dict]:
        changes = []
        users = (
            self.online_session.query(User)
            .filter(User.updated_at > self.last_sync_timestamp)
            .all()
        )
        embeddings = (
            self.online_session.query(Embedding)
            .filter(Embedding.updated_at > self.last_sync_timestamp)
            .all()
        )

        for user in users:
            user_dict = {
                k: v for k, v in user.__dict__.items() if k != "_sa_instance_state"
            }
            changes.append({"table": "users", "data": user_dict})
        for embedding in embeddings:
            embedding_dict = {
                k: v for k, v in embedding.__dict__.items() if k != "_sa_instance_state"
            }
            changes.append({"table": "embeddings", "data": embedding_dict})

        return changes

    async def apply_changes_to_offline(self, changes: List[Dict]):
        for change in changes:
            if change["table"] == "users":
                self.update_or_insert_user(change["data"])
            elif change["table"] == "embeddings":
                self.update_or_insert_embedding(change["data"])
        self.offline_session.commit()

    def update_or_insert_user(self, user_data: Dict):
        user = (
            self.offline_session.query(User)
            .filter_by(user_id=user_data["user_id"])
            .first()
        )
        if user:
            logger.info(f"Updating existing user: {user_data['user_id']}")
            for key, value in user_data.items():
                if key != "_sa_instance_state":
                    setattr(user, key, value)
        else:
            logger.info(f"Inserting new user: {user_data['user_id']}")
            user = User(
                **{k: v for k, v in user_data.items() if k != "_sa_instance_state"}
            )
            self.offline_session.add(user)

    def update_or_insert_embedding(self, embedding_data: Dict):
        embedding = (
            self.offline_session.query(Embedding)
            .filter_by(user_id=embedding_data["user_id"])
            .first()
        )
        if embedding:
            logger.info(
                f"Updating existing embedding for user: {embedding_data['user_id']}"
            )
            for key, value in embedding_data.items():
                if key != "_sa_instance_state":
                    setattr(embedding, key, value)
        else:
            logger.info(
                f"Inserting new embedding for user: {embedding_data['user_id']}"
            )
            embedding = Embedding(
                **{k: v for k, v in embedding_data.items() if k != "_sa_instance_state"}
            )
            self.offline_session.add(embedding)


async def simulate_fr_online_inserts(db_url: str, insert_interval: int = 10):
    engine = create_engine(db_url)

    # Create vector extension if it doesn't exist
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info(f"Ensured vector extension exists for {engine.url}")

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    insert_count = 0
    while True:
        session = Session()
        try:
            user_id = f"user_{random.randint(1000, 9999)}"
            user = User(user_id=user_id, name=f"User {user_id}")
            session.add(user)

            embedding = Embedding(
                user_id=user_id, embedding=[random.random() for _ in range(512)]
            )
            session.add(embedding)

            session.commit()
            insert_count += 1
            logger.info(
                f"Inserted new user and embedding: {user_id} (Total inserts: {insert_count})"
            )
        except Exception as e:
            logger.error(f"Error inserting data: {str(e)}")
        finally:
            session.close()

        await asyncio.sleep(insert_interval)


async def main():
    fr_online_db_url = "postgresql://fr_user:fr_password@fr-online-db:5432/fr_online"
    fr_offline_db_url = "postgresql://fr_user:fr_password@fr-offline-db:5432/fr_offline"

    logger.info("Starting FR Sync Simulation")
    logger.info(f"Online DB URL: {fr_online_db_url}")
    logger.info(f"Offline DB URL: {fr_offline_db_url}")

    sync_service = FROfflineSyncService(
        fr_online_db_url, fr_offline_db_url, sync_interval=10
    )

    await asyncio.gather(
        simulate_fr_online_inserts(fr_online_db_url, insert_interval=5),
        sync_service.start(),
    )


if __name__ == "__main__":
    asyncio.run(main())
