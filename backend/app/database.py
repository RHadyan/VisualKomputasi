import aiosqlite
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "predictions.db")


async def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                visual_score REAL,
                text_score REAL,
                hybrid_score REAL,
                heatmap_path TEXT,
                image_path TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await db.commit()


async def save_prediction(
    filename: str,
    label: str,
    confidence: float,
    visual_score: float = None,
    text_score: float = None,
    hybrid_score: float = None,
    heatmap_path: str = None,
    image_path: str = None,
):
    """Save a prediction result to the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO predictions (filename, label, confidence, visual_score, text_score, hybrid_score, heatmap_path, image_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                label,
                confidence,
                visual_score,
                text_score,
                hybrid_score,
                heatmap_path,
                image_path,
                datetime.now().isoformat(),
            ),
        )
        await db.commit()
        cursor = await db.execute("SELECT last_insert_rowid()")
        row = await cursor.fetchone()
        return row[0]


async def get_all_predictions(limit: int = 50, offset: int = 0):
    """Get all predictions from the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_prediction_by_id(prediction_id: int):
    """Get a single prediction by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def delete_prediction(prediction_id: int):
    """Delete a prediction by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        await db.commit()
        return True


async def get_prediction_count():
    """Get total number of predictions."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM predictions")
        row = await cursor.fetchone()
        return row[0]
