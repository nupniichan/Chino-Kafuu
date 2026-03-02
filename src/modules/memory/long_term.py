import logging
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from src.setting import LONG_TERM_DB_PATH

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Manages conversation summaries in SQLite database."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with database path."""
        self.db_path = db_path or LONG_TERM_DB_PATH
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self):
        """Create database directory if not exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_database(self):
        """Create tables if not exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        summary TEXT NOT NULL,
                        original_messages TEXT,
                        message_count INTEGER,
                        importance_score REAL DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_importance 
                    ON conversation_summaries(importance_score)
                """)
                
                conn.commit()
                logger.info(f"Long-term memory database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def add_summary(
        self,
        summary: str,
        original_messages: List[Dict[str, Any]],
        importance_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add conversation summary to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO conversation_summaries 
                    (summary, original_messages, message_count, importance_score, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    summary,
                    json.dumps(original_messages, ensure_ascii=False),
                    len(original_messages),
                    importance_score,
                    json.dumps(metadata or {}, ensure_ascii=False)
                ))
                
                conn.commit()
                summary_id = cursor.lastrowid
                
                logger.info(f"Saved summary {summary_id}")
                return summary_id
                
        except Exception as e:
            logger.error(f"Failed to add summary: {e}")
            raise
    
    def get_recent_summaries(
        self,
        limit: int = 5,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve recent summaries across all sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM conversation_summaries 
                    WHERE importance_score >= ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (min_importance, limit))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to retrieve recent summaries: {e}")
            return []
    
    def get_summary_count(self, min_importance: float = 0.0) -> int:
        """Get total count of summaries without loading them."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM conversation_summaries WHERE importance_score >= ?",
                    (min_importance,)
                )
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count summaries: {e}")
            return 0

    def get_high_importance_summaries(
        self,
        min_score: float = 0.8,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve high-importance summaries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM conversation_summaries 
                    WHERE importance_score >= ?
                    ORDER BY importance_score DESC, created_at DESC
                    LIMIT ?
                """, (min_score, limit))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to retrieve high-importance summaries: {e}")
            return []
    
    def update_importance_score(self, summary_id: int, score: float):
        """Update importance score for a summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE conversation_summaries 
                    SET importance_score = ?
                    WHERE id = ?
                """, (score, summary_id))
                
                conn.commit()
                logger.info(f"Updated importance score for summary {summary_id}: {score}")
                
        except Exception as e:
            logger.error(f"Failed to update importance score: {e}")
            raise
    
    def delete_summary(self, summary_id: int) -> bool:
        """Delete a specific summary by ID. Returns True if deleted."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM conversation_summaries WHERE id = ?",
                    (summary_id,)
                )
                conn.commit()
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Deleted summary {summary_id}")
                return deleted
        except Exception as e:
            logger.error(f"Failed to delete summary {summary_id}: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about long-term memory storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COALESCE(AVG(importance_score), 0) as avg_importance,
                        COALESCE(SUM(message_count), 0) as total_messages,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest
                    FROM conversation_summaries
                """)
                row = cursor.fetchone()

                return {
                    "total_summaries": row[0],
                    "avg_importance": round(row[1], 3),
                    "total_messages_summarized": row[2],
                    "oldest_summary": row[3],
                    "newest_summary": row[4],
                    "db_path": self.db_path
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_summaries": 0, "avg_importance": 0, "total_messages_summarized": 0}

    def clear_old_summaries(self, days: int = 30):
        """Clear summaries older than specified days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM conversation_summaries 
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                    AND importance_score < 0.8
                """, (days,))
                
                deleted = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleared {deleted} old summaries")
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to clear old summaries: {e}")
            return 0