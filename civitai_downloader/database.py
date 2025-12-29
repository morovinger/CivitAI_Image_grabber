"""SQLite database operations for tracking downloaded images."""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Tuple

from .config import DATABASE_FILENAME, SCRIPT_DIR


class NullTracker:
    """No-op tracker that skips all database operations."""
    
    def __init__(self):
        """Initialize null tracker."""
        self.logger = logging.getLogger('CivitaiDownloader')
        self.logger.info("Database tracking disabled (NullTracker active)")
    
    def is_image_tracked(self, image_id: str, quality: str) -> bool:
        """Always return False - never tracked."""
        return False
    
    def mark_image_downloaded(
        self,
        image_id: str,
        path: str,
        quality: str,
        tags: Optional[List[str]] = None,
        url: Optional[str] = None,
        checkpoint_name: Optional[str] = None
    ) -> bool:
        """No-op - don't track anything."""
        return True
    
    def get_images_by_tag(self, tag: str) -> List[Tuple[str, str, str]]:
        """Return empty list."""
        return []
    
    def get_download_count(self) -> int:
        """Return 0."""
        return 0
    
    def close(self) -> None:
        """No-op."""
        pass


class ImageTracker:
    """Handles SQLite database operations for tracking downloaded images."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. Defaults to script directory.
        """
        self.logger = logging.getLogger('CivitaiDownloader')
        self.db_path = db_path or os.path.join(SCRIPT_DIR, DATABASE_FILENAME)
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()
    
    def _init_db(self) -> None:
        """Initializes the SQLite DB connection and creates tables if needed."""
        try:
            self.conn = sqlite3.connect(self.db_path, timeout=10)
            self.logger.info(f"Connected to tracking database: {self.db_path}")
            
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            
            # Create tracked_images table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracked_images (
                    image_key TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    path TEXT NOT NULL,
                    download_date TEXT NOT NULL,
                    url TEXT,
                    checkpoint_name TEXT
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracked_images_key ON tracked_images (image_key)')
            
            # Create image_tags table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS image_tags (
                    image_key TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (image_key, tag),
                    FOREIGN KEY(image_key) REFERENCES tracked_images(image_key) ON DELETE CASCADE
                ) WITHOUT ROWID;
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags (tag)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_tags_key ON image_tags (image_key)')
            
            self.conn.commit()
            self.logger.debug("Database schema initialized successfully.")
            
        except sqlite3.Error as e:
            self.logger.critical(f"Database error during initialization: {e}", exc_info=True)
            raise
    
    def is_image_tracked(self, image_id: str, quality: str) -> bool:
        """Check if an image with given ID and quality is already tracked.
        
        Args:
            image_id: The CivitAI image ID
            quality: Quality level (SD or HD)
        
        Returns:
            True if image is already tracked
        """
        if not self.conn:
            return False
        
        image_key = f"{image_id}_{quality}"
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM tracked_images WHERE image_key = ?", (image_key,))
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            self.logger.error(f"Database error checking tracked status: {e}")
            return False
    
    def mark_image_downloaded(
        self,
        image_id: str,
        path: str,
        quality: str,
        tags: Optional[List[str]] = None,
        url: Optional[str] = None,
        checkpoint_name: Optional[str] = None
    ) -> bool:
        """Mark an image as downloaded in the database.
        
        Args:
            image_id: The CivitAI image ID
            path: Local file path where image was saved
            quality: Quality level (SD or HD)
            tags: Optional list of tags associated with this download
            url: Optional original URL
            checkpoint_name: Optional checkpoint/model name used
        
        Returns:
            True if successfully recorded
        """
        if not self.conn:
            return False
        
        image_key = f"{image_id}_{quality}"
        download_date = datetime.utcnow().isoformat()
        
        try:
            cursor = self.conn.cursor()
            
            # Insert or replace the image record
            cursor.execute('''
                INSERT OR REPLACE INTO tracked_images 
                (image_key, image_id, quality, path, download_date, url, checkpoint_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (image_key, image_id, quality, path, download_date, url, checkpoint_name))
            
            # Insert tags if provided
            if tags:
                for tag in tags:
                    if tag:
                        cursor.execute('''
                            INSERT OR IGNORE INTO image_tags (image_key, tag)
                            VALUES (?, ?)
                        ''', (image_key, tag))
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error marking image downloaded: {e}")
            return False
    
    def get_images_by_tag(self, tag: str) -> List[Tuple[str, str, str]]:
        """Get all images associated with a specific tag.
        
        Args:
            tag: The tag to search for
        
        Returns:
            List of tuples (image_id, path, quality)
        """
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT ti.image_id, ti.path, ti.quality
                FROM tracked_images ti
                JOIN image_tags it ON ti.image_key = it.image_key
                WHERE it.tag = ?
            ''', (tag,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            self.logger.error(f"Database error getting images by tag: {e}")
            return []
    
    def get_download_count(self) -> int:
        """Get total number of tracked downloads."""
        if not self.conn:
            return 0
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracked_images")
            result = cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.Error as e:
            self.logger.error(f"Database error getting count: {e}")
            return 0
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.logger.info("Database connection closed.")
            except sqlite3.Error as e:
                self.logger.error(f"Error closing database: {e}")
            finally:
                self.conn = None


