import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
try:
    from .config import DB_PATH
    from .logger import get_logger
except ImportError:
    from config import DB_PATH
    from logger import get_logger

logger = get_logger(__name__)

class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the database schema if it doesn't exist"""
        with self.connection() as conn:
            # Table for mapping PDF names to correct web app names
            conn.execute("""
                CREATE TABLE IF NOT EXISTS name_aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_name TEXT UNIQUE,
                    web_name TEXT,
                    confidence REAL DEFAULT 1.0,
                    confirmed INTEGER DEFAULT 1,
                    source TEXT DEFAULT 'manual',
                    created_at TEXT,
                    used_count INTEGER DEFAULT 0
                )
            """)

            # Table for logging each processing run
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_file TEXT,
                    engine TEXT,
                    total INTEGER,
                    matched INTEGER,
                    manual INTEGER,
                    unresolved INTEGER,
                    started_at TEXT,
                    finished_at TEXT
                )
            """)

            # Table for names that couldn't be resolved or were skipped
            conn.execute("""
                CREATE TABLE IF NOT EXISTS unresolved_names (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_name TEXT UNIQUE,
                    seen_count INTEGER DEFAULT 1,
                    last_seen TEXT,
                    notes TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT,
                    name TEXT,
                    municipality TEXT,
                    place_norm TEXT,
                    filijala TEXT
                )
            """)
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_customers_filijala
                             ON customers(UPPER(filijala))
                         """)
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_customers_place_norm
                             ON customers(place_norm)
                         """)
            # Migracija: kolona filijala u name_aliases (za stare baze)
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(name_aliases)")]
            if "filijala" not in cols:
                conn.execute("ALTER TABLE name_aliases ADD COLUMN filijala TEXT")

            logger.debug("Database initialized.")

    @contextmanager
    def connection(self):
        """
            Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def get_alias(self, pdf_name: str) -> dict | None:
        """Retrieves an alias for a given PDF name and increments used_count"""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM name_aliases WHERE pdf_name = ?",
                (pdf_name,)
            ).fetchone()

            if row:
                conn.execute(
                    "UPDATE name_aliases SET used_count = used_count + 1 WHERE id = ?",
                    (row['id'],)
                )
                # Fetch updated row to include incremented used_count
                row = conn.execute(
                    "SELECT * FROM name_aliases WHERE id = ?",
                    (row['id'],)
                ).fetchone()
                return dict(row)
        return None

    def save_alias(self, pdf_name, web_name, confidence=1.0, confirmed=True, source="manual", filijala=None) -> bool:
        """Saves or updates an alias for a given PDF name (Upsert)"""
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO name_aliases (pdf_name, web_name, confidence, confirmed, source, created_at, filijala)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pdf_name) DO UPDATE SET
                    web_name = excluded.web_name,
                    confidence = excluded.confidence,
                    confirmed = excluded.confirmed,
                    source = excluded.source,
                    filijala = excluded.filijala
            """, (pdf_name, web_name, confidence, 1 if confirmed else 0, source, now, filijala or None))
        return True

    def get_all_aliases(self) -> list[dict]:
        """Returns all saved aliases"""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM name_aliases ORDER BY pdf_name").fetchall()
            return [dict(r) for r in rows]

    def start_run(self, pdf_file: str, engine: str) -> int:
        """Logs the start of a processing run and returns the run ID"""
        now = datetime.now().isoformat()
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO processing_log (pdf_file, engine, started_at) VALUES (?, ?, ?)",
                (pdf_file, engine, now)
            )
            return cursor.lastrowid

    def finish_run(self, run_id: int, total: int, matched: int, manual: int, unresolved: int):
        """Updates the log entry for a finished processing run"""
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute("""
                UPDATE processing_log 
                SET total = ?, matched = ?, manual = ?, unresolved = ?, finished_at = ?
                WHERE id = ?
            """, (total, matched, manual, unresolved, now, run_id))

    def add_unresolved(self, pdf_name, notes=""):
        """Adds a name to the unresolved list or increments its seen count"""
        now = datetime.now().isoformat()
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO unresolved_names (pdf_name, seen_count, last_seen, notes)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(pdf_name) DO UPDATE SET
                    seen_count = unresolved_names.seen_count + 1,
                    last_seen = excluded.last_seen,
                    notes = excluded.notes
            """, (pdf_name, now, notes))

    def get_unresolved(self) -> list[dict]:
        """Returns all unresolved names"""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM unresolved_names ORDER BY seen_count DESC").fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Returns overall database statistics"""
        with self.connection() as conn:
            total_aliases = conn.execute("SELECT COUNT(*) FROM name_aliases").fetchone()[0]
            confirmed_aliases = conn.execute("SELECT COUNT(*) FROM name_aliases WHERE confirmed = 1").fetchone()[0]
            total_runs = conn.execute("SELECT COUNT(*) FROM processing_log").fetchone()[0]

            log_stats = conn.execute("""
                SELECT SUM(total), SUM(matched) FROM processing_log
            """).fetchone()
            total_processed = log_stats[0] or 0
            total_matched = log_stats[1] or 0

            unresolved_count = conn.execute("SELECT COUNT(*) FROM unresolved_names").fetchone()[0]

            accuracy_pct = (total_matched / total_processed * 100) if total_processed > 0 else 0

            return {
                "total_aliases": total_aliases,
                "confirmed_aliases": confirmed_aliases,
                "total_runs": total_runs,
                "total_processed": total_processed,
                "total_matched": total_matched,
                "accuracy_pct": round(accuracy_pct, 2),
                "unresolved_count": unresolved_count
            }
