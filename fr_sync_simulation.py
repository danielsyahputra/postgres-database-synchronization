import asyncio
import json
import logging
import random
import time
from datetime import datetime
from typing import Dict, List

import asyncpg
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, String, create_engine, text
from sqlalchemy.exc import OperationalError
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
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    embedding = relationship(
        "Embedding", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class Embedding(Base):
    __tablename__ = "embeddings"
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    embedding = Column(Vector(512))
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    user = relationship("User", back_populates="embedding")


class FROfflineSyncService:
    def __init__(
        self, fr_online_db_url: str, fr_offline_db_url: str, sync_interval: int = 60
    ):
        self.fr_online_db_url = fr_online_db_url
        self.fr_offline_db_url = fr_offline_db_url
        self.online_db_name = "ONLINE"
        self.offline_db_name = "OFFLINE"
        self.sync_interval = sync_interval
        self.last_sync_timestamp = datetime.min
        self.online_engine = self.create_engine_with_retry(fr_online_db_url)
        self.offline_engine = self.create_engine_with_retry(fr_offline_db_url)
        self.create_vector_extension(self.online_engine)
        self.create_vector_extension(self.offline_engine)
        Base.metadata.create_all(self.online_engine)
        Base.metadata.create_all(self.offline_engine)
        self.online_session = sessionmaker(bind=self.online_engine)()
        self.offline_session = sessionmaker(bind=self.offline_engine)()
        self.is_offline = False
        self.notification_queue = asyncio.Queue()

    def create_engine_with_retry(self, db_url, max_retries=5, retry_interval=5):
        for attempt in range(max_retries):
            try:
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info(
                    f"[{self.online_db_name if 'online' in db_url else self.offline_db_name}] Successfully connected to database: {db_url}"
                )
                return engine
            except OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[{self.online_db_name if 'online' in db_url else self.offline_db_name}] Connection attempt {attempt + 1} failed. Retrying in {retry_interval} seconds..."
                    )
                    time.sleep(retry_interval)
                else:
                    logger.error(
                        f"[{self.online_db_name if 'online' in db_url else self.offline_db_name}] Failed to connect to database after {max_retries} attempts: {e}"
                    )
                    raise

    def create_vector_extension(self, engine):
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        logger.info(
            f"[{self.online_db_name if 'online' in str(engine.url) else self.offline_db_name}] Ensured vector extension exists for {engine.url}"
        )

    async def start(self):
        await asyncio.gather(
            self.run_periodic_sync(),
            self.listen_for_changes(),
            self.process_notifications(),
            self.simulate_offline_period(),
        )

    async def run_periodic_sync(self):
        while True:
            if not self.is_offline:
                await self.sync_databases()
            await asyncio.sleep(self.sync_interval)

    async def listen_for_changes(self):
        while True:
            try:
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
                logger.info(
                    f"[{self.online_db_name}] Listening for changes on the online database"
                )
                while True:
                    try:
                        await asyncio.sleep(1)
                    except asyncio.CancelledError:
                        await conn.close()
                        raise
            except asyncpg.exceptions.ConnectionDoesNotExistError:
                logger.error(
                    f"[{self.online_db_name}] Connection lost. Attempting to reconnect..."
                )
                await asyncio.sleep(5)

    async def handle_change_notification(self, connection, pid, channel, payload):
        if not self.is_offline:
            logger.info(
                f"[{self.online_db_name}] Change detected: {json.loads(payload)['data']['user_id']}"
            )
            await self.notification_queue.put(json.loads(payload))

    async def process_notifications(self):
        while True:
            if not self.is_offline:
                try:
                    notification = await asyncio.wait_for(
                        self.notification_queue.get(), timeout=1.0
                    )
                    await self.apply_single_change(notification)
                except asyncio.TimeoutError:
                    pass  # No notification received within the timeout
            else:
                await asyncio.sleep(1)  # Sleep while offline to reduce CPU usage

    async def apply_single_change(self, notification):
        table = notification["table"]
        action = notification["action"]
        data = notification["data"]

        if table == "users":
            if action in ["INSERT", "UPDATE"]:
                self.update_or_insert_user(data)
            elif action == "DELETE":
                self.delete_user(data["user_id"])
        elif table == "embeddings":
            if action in ["INSERT", "UPDATE"]:
                self.update_or_insert_embedding(data)
            elif action == "DELETE":
                self.delete_embedding(data["user_id"])

        self.offline_session.commit()
        logger.info(
            f"[{self.offline_db_name}] Applied {action} on {table} for user_id: {data['user_id']}"
        )

    async def sync_databases(self):
        try:
            logger.info(f"[{self.online_db_name}] Starting database synchronization")
            changes = self.get_changes_since_last_sync()
            if changes:
                logger.info(
                    f"[{self.online_db_name}] Found {len(changes)} changes to synchronize"
                )
                await self.apply_changes_to_offline(changes)
                logger.info(
                    f"[{self.offline_db_name}] Applied {len(changes)} changes to the offline database"
                )
            else:
                logger.info(f"[{self.online_db_name}] No changes found to synchronize")
            self.last_sync_timestamp = datetime.now()
            logger.info(
                f"[{self.offline_db_name}] Sync completed at {self.last_sync_timestamp}"
            )
        except Exception as e:
            logger.error(f"[SYNC] Error during synchronization: {str(e)}")

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
            embedding_dict["embedding"] = embedding_dict["embedding"].tolist()
            changes.append({"table": "embeddings", "data": embedding_dict})
        logger.info(
            f"[{self.online_db_name}] Fetched {len(changes)} changes since last sync"
        )
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
            logger.info(
                f"[{self.offline_db_name}] Updating existing user: {user_data['user_id']}"
            )
            for key, value in user_data.items():
                if key != "_sa_instance_state":
                    setattr(user, key, value)
        else:
            logger.info(
                f"[{self.offline_db_name}] Inserting new user: {user_data['user_id']}"
            )
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
                f"[{self.offline_db_name}] Updating existing embedding for user: {embedding_data['user_id']}"
            )
            for key, value in embedding_data.items():
                if key == "embedding":
                    value = self.process_embedding_data(value)
                if key != "_sa_instance_state":
                    setattr(embedding, key, value)
        else:
            logger.info(
                f"[{self.offline_db_name}] Inserting new embedding for user: {embedding_data['user_id']}"
            )
            embedding_data["embedding"] = self.process_embedding_data(
                embedding_data["embedding"]
            )
            embedding = Embedding(
                **{k: v for k, v in embedding_data.items() if k != "_sa_instance_state"}
            )
            self.offline_session.add(embedding)

    def process_embedding_data(self, embedding_data):
        if isinstance(embedding_data, str):
            try:
                embedding_data = json.loads(embedding_data)
            except json.JSONDecodeError:
                logger.error(
                    f"[{self.offline_db_name}] Failed to parse embedding data: {embedding_data}"
                )
                return [0.0] * 512

        if isinstance(embedding_data, list):
            return [float(x) for x in embedding_data]
        else:
            logger.error(
                f"[{self.offline_db_name}] Unexpected embedding data type: {type(embedding_data)}"
            )
            return [0.0] * 512

    def delete_user(self, user_id):
        user = self.offline_session.query(User).filter_by(user_id=user_id).first()
        if user:
            self.offline_session.delete(user)
            logger.info(f"[{self.offline_db_name}] Deleted user: {user_id}")

    def delete_embedding(self, user_id):
        embedding = (
            self.offline_session.query(Embedding).filter_by(user_id=user_id).first()
        )
        if embedding:
            self.offline_session.delete(embedding)
            logger.info(
                f"[{self.offline_db_name}] Deleted embedding for user: {user_id}"
            )

    async def simulate_offline_period(self):
        await asyncio.sleep(60)  # Wait for 60 seconds before simulating offline period
        logger.info(
            f"[{self.offline_db_name}] Simulating offline period for 30 seconds"
        )
        self.is_offline = True
        await asyncio.sleep(30)
        self.is_offline = False
        logger.info(f"[{self.offline_db_name}] Offline period ended")
        await self.sync_databases()  # Perform a sync immediately after coming back online


async def simulate_fr_online_inserts(db_url: str, insert_interval: int = 5):
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info(f"[ONLINE] Ensured vector extension exists for {engine.url}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    insert_count = 0
    while True:
        session = Session()
        try:
            user_id = f"user_{random.randint(1000, 9999)}"
            user = User(user_id=user_id, name=f"User {user_id}")
            session.add(user)
            embedding_vector = [random.uniform(-1, 1) for _ in range(512)]
            embedding = Embedding(user_id=user_id, embedding=embedding_vector)
            session.add(embedding)
            session.commit()
            insert_count += 1
            logger.info(
                f"[ONLINE] Inserted new user and embedding: {user_id} (Total inserts: {insert_count})"
            )
        except Exception as e:
            logger.error(f"[ONLINE] Error inserting data: {str(e)}")
        finally:
            session.close()
        await asyncio.sleep(insert_interval)


async def main():
    fr_online_db_url = "postgresql://fr_user:fr_password@fr-online-db:5432/fr_online"
    fr_offline_db_url = "postgresql://fr_user:fr_password@fr-offline-db:5432/fr_offline"
    logger.info("[MAIN] Starting FR Sync Simulation")
    logger.info(f"[MAIN] Online DB URL: {fr_online_db_url}")
    logger.info(f"[MAIN] Offline DB URL: {fr_offline_db_url}")
    sync_service = FROfflineSyncService(
        fr_online_db_url, fr_offline_db_url, sync_interval=15
    )
    await asyncio.gather(
        simulate_fr_online_inserts(fr_online_db_url, insert_interval=5),
        sync_service.start(),
    )


if __name__ == "__main__":
    asyncio.run(main())
