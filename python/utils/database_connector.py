#!/usr/bin/env python3
"""
Database Connector Module for ML MCP System
Connect to various databases for data loading
"""

import pandas as pd
import json
import sys
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# PostgreSQL
try:
    import psycopg2
    from psycopg2 import sql
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

# MySQL
try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    pymysql = None

# MongoDB
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoClient = None

# SQLite (built-in)
import sqlite3
SQLITE_AVAILABLE = True


class PostgreSQLConnector:
    """PostgreSQL database connector"""

    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str):
        """
        Initialize PostgreSQL connector

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 required. Install with: pip install psycopg2-binary")

        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None

    def connect(self) -> Dict[str, Any]:
        """Establish connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            return {'success': True, 'message': 'Connected to PostgreSQL'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def query(self, query: str) -> pd.DataFrame:
        """
        Execute query and return DataFrame

        Args:
            query: SQL query

        Returns:
            Query results as DataFrame
        """
        if self.conn is None:
            self.connect()

        return pd.read_sql_query(query, self.conn)

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute non-SELECT query

        Args:
            query: SQL query

        Returns:
            Execution result
        """
        if self.conn is None:
            self.connect()

        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()

            return {
                'success': True,
                'rows_affected': cursor.rowcount
            }

        except Exception as e:
            self.conn.rollback()
            return {'success': False, 'error': str(e)}

    def list_tables(self) -> List[str]:
        """List all tables"""
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """
        df = self.query(query)
        return df['table_name'].tolist()

    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()


class MySQLConnector:
    """MySQL database connector"""

    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str):
        """
        Initialize MySQL connector

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
        """
        if not MYSQL_AVAILABLE:
            raise ImportError("pymysql required. Install with: pip install pymysql")

        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None

    def connect(self) -> Dict[str, Any]:
        """Establish connection"""
        try:
            self.conn = pymysql.connect(**self.connection_params)
            return {'success': True, 'message': 'Connected to MySQL'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def query(self, query: str) -> pd.DataFrame:
        """
        Execute query and return DataFrame

        Args:
            query: SQL query

        Returns:
            Query results as DataFrame
        """
        if self.conn is None:
            self.connect()

        return pd.read_sql_query(query, self.conn)

    def list_tables(self) -> List[str]:
        """List all tables"""
        query = "SHOW TABLES"
        df = self.query(query)
        return df.iloc[:, 0].tolist()

    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()


class MongoDBConnector:
    """MongoDB database connector"""

    def __init__(self, host: str, port: int, database: str,
                 username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize MongoDB connector

        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Username (optional)
            password: Password (optional)
        """
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo required. Install with: pip install pymongo")

        if username and password:
            connection_string = f"mongodb://{username}:{password}@{host}:{port}/"
        else:
            connection_string = f"mongodb://{host}:{port}/"

        self.client = MongoClient(connection_string)
        self.db = self.client[database]

    def find(self, collection: str, query: Optional[Dict] = None,
            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Query collection and return DataFrame

        Args:
            collection: Collection name
            query: MongoDB query (None = all documents)
            limit: Maximum documents to return

        Returns:
            Query results as DataFrame
        """
        query = query or {}
        cursor = self.db[collection].find(query)

        if limit:
            cursor = cursor.limit(limit)

        documents = list(cursor)
        return pd.DataFrame(documents)

    def list_collections(self) -> List[str]:
        """List all collections"""
        return self.db.list_collection_names()

    def close(self):
        """Close connection"""
        self.client.close()


class SQLiteConnector:
    """SQLite database connector"""

    def __init__(self, database_path: str):
        """
        Initialize SQLite connector

        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = database_path
        self.conn = None

    def connect(self) -> Dict[str, Any]:
        """Establish connection"""
        try:
            self.conn = sqlite3.connect(self.database_path)
            return {'success': True, 'message': 'Connected to SQLite'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def query(self, query: str) -> pd.DataFrame:
        """
        Execute query and return DataFrame

        Args:
            query: SQL query

        Returns:
            Query results as DataFrame
        """
        if self.conn is None:
            self.connect()

        return pd.read_sql_query(query, self.conn)

    def list_tables(self) -> List[str]:
        """List all tables"""
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        df = self.query(query)
        return df['name'].tolist()

    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()


class DatabaseFactory:
    """Factory for creating database connectors"""

    @staticmethod
    def create(db_type: str, **kwargs):
        """
        Create database connector

        Args:
            db_type: 'postgresql', 'mysql', 'mongodb', or 'sqlite'
            **kwargs: Database-specific connection parameters

        Returns:
            Database connector instance
        """
        if db_type == 'postgresql':
            return PostgreSQLConnector(
                host=kwargs['host'],
                port=kwargs.get('port', 5432),
                database=kwargs['database'],
                user=kwargs['user'],
                password=kwargs['password']
            )

        elif db_type == 'mysql':
            return MySQLConnector(
                host=kwargs['host'],
                port=kwargs.get('port', 3306),
                database=kwargs['database'],
                user=kwargs['user'],
                password=kwargs['password']
            )

        elif db_type == 'mongodb':
            return MongoDBConnector(
                host=kwargs['host'],
                port=kwargs.get('port', 27017),
                database=kwargs['database'],
                username=kwargs.get('username'),
                password=kwargs.get('password')
            )

        elif db_type == 'sqlite':
            return SQLiteConnector(database_path=kwargs['database_path'])

        else:
            raise ValueError(f"Unknown database type: {db_type}")

    @staticmethod
    def get_available_databases() -> Dict[str, bool]:
        """Get available database connectors"""
        return {
            'postgresql': POSTGRES_AVAILABLE,
            'mysql': MYSQL_AVAILABLE,
            'mongodb': MONGODB_AVAILABLE,
            'sqlite': SQLITE_AVAILABLE
        }


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python database_connector.py <action>")
        print("Actions: check_availability")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'check_availability':
            databases = DatabaseFactory.get_available_databases()

            result = {
                'databases': databases,
                'install_commands': {
                    'postgresql': 'pip install psycopg2-binary',
                    'mysql': 'pip install pymysql',
                    'mongodb': 'pip install pymongo',
                    'sqlite': 'built-in'
                }
            }

        else:
            result = {'error': f'Unknown action: {action}'}

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()