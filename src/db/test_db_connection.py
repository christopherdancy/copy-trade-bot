from .database import DatabaseConnection
import os
from urllib.parse import urlparse
import psycopg2
from sqlalchemy import text

class DatabaseTester:
    def __init__(self):
        self.db_url = os.getenv('DB_URL')
        self.parsed = urlparse(self.db_url)
        self.db = None

    def test_direct_connection(self):
        """Test direct database connection"""
        print("\nTesting direct database connection...")
        
        try:
            conn = psycopg2.connect(
                dbname="postgres",
                user="postgres",
                password=self.parsed.password,
                host=self.parsed.hostname,
                port=5432
            )
            print("✅ Direct connection successful!")
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False

    def test_sqlalchemy_operations(self):
        """Test SQLAlchemy operations"""
        print("\nTesting SQLAlchemy operations...")
        
        try:
            if not self.db:
                self.db = DatabaseConnection()
                
            # Test 1: Basic Query
            session = self.db.get_session()
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1
            print("✅ Basic query successful")
            session.close()
            
            # Test 2: Transaction
            session = self.db.get_session()
            try:
                with session.begin():
                    result = session.execute(text("SELECT 2")).scalar()
                    assert result == 2
                print("✅ Transaction successful")
            finally:
                session.close()
            
            print("✅ Session management successful")
            return True
            
        except Exception as e:
            print(f"❌ Operation failed: {str(e)}")
            if 'session' in locals():
                session.close()
            return False

def run_tests():
    tester = DatabaseTester()
    
    # Run connection test first
    if not tester.test_direct_connection():
        print("\n❌ Basic connection failed - skipping further tests")
        return False
        
    # Run SQLAlchemy operations test
    if not tester.test_sqlalchemy_operations():
        print("\n❌ SQLAlchemy operations failed")
        return False
        
    print("\n✅ All database tests passed successfully!")
    return True

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 