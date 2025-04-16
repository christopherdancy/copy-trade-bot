from .database import DatabaseConnection
from .models import Base  # This will import all models

if __name__ == "__main__":
    db = DatabaseConnection()
    Base.metadata.create_all(db.engine)
    print("Tables created successfully") 