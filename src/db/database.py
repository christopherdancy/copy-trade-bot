import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
import logging

# Load environment variables
load_dotenv()

Base = declarative_base()

class DatabaseConnection:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.engine = self._create_engine()
        self.Session = self._create_session()

    def _create_engine(self):
        try:
            connection_string = os.getenv('DB_URL')
            if not connection_string:
                raise ValueError("DB_URL not found in environment variables")
                
            return create_engine(
                connection_string,
                pool_size=5,                # Connection pool size
                max_overflow=10,            # Max extra connections
                pool_timeout=30,            # Seconds to wait for connection
                pool_recycle=1800,          # Recycle connections after 30 mins
                echo=False                  # Set to True for SQL logging
            )
        except Exception as e:
            self.logger.error(f"Failed to create database engine: {str(e)}")
            raise

    def _create_session(self):
        """Create a scoped session factory"""
        return scoped_session(sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        ))

    def get_session(self):
        """Get a new database session"""
        return self.Session()

    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                # Execute a simple query to verify connection
                conn.execute("SELECT 1")
                self.logger.info("Successfully connected to the database")
                return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            return False

    def init_db(self):
        """Initialize database tables"""
        try:
            Base.metadata.create_all(self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {str(e)}")
            raise 