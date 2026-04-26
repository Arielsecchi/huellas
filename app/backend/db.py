"""Capa SQLite para los intentos de practica.

Schema unico:

  attempts(
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,                -- ISO 8601 UTC
    klass_asked   TEXT NOT NULL,                -- "A" | "I" | "E" | "V"
    source        TEXT NOT NULL,                -- "gan" | "real"
    klass_answered TEXT,                        -- NULL si pendiente
    correct       INTEGER                       -- 0/1, NULL si pendiente
  )

Flujo:
  1. POST/GET /api/sample -> insert (klass_asked, source, ts), answered=NULL.
     Devuelve id + png al cliente.
  2. POST /api/answer -> update answered + correct para ese id.
  3. GET /api/stats -> agrega solo filas con answered != NULL.

Limpieza: filas pendientes (answered=NULL) > 24h se purgan al insertar.
SQLite con `check_same_thread=False` + un Lock en el modulo asi varios
workers de uvicorn no se pisan (en realidad uvicorn dev corre 1 worker
solo, pero esto deja el camino abierto).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

# PROJECT_ROOT = huellas-gan/. La DB vive en app/backend/data/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "app" / "backend" / "data" / "stats.db"

PENDING_TTL_HOURS = 24

_lock = threading.Lock()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = _connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS attempts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            klass_asked     TEXT NOT NULL,
            source          TEXT NOT NULL,
            klass_answered  TEXT,
            correct         INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_attempts_answered ON attempts(klass_answered);
        CREATE INDEX IF NOT EXISTS idx_attempts_ts ON attempts(ts);
    """)
    conn.commit()
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def insert_pending(conn: sqlite3.Connection,
                   klass_asked: str,
                   source: str) -> int:
    """Inserta un intento pendiente (sin respuesta aun). Devuelve el id."""
    with _lock:
        cur = conn.execute(
            "INSERT INTO attempts (ts, klass_asked, source) VALUES (?, ?, ?)",
            (_now_iso(), klass_asked, source),
        )
        conn.commit()
        return int(cur.lastrowid)


def record_answer(conn: sqlite3.Connection,
                  attempt_id: int,
                  klass_answered: str) -> dict | None:
    """Updatea el intento. Devuelve la row final o None si no existia.

    Si el intento ya tenia respuesta previa (intento de re-submit), se
    devuelve la row existente sin tocarla — un attempt vale solo una vez.
    """
    with _lock:
        row = conn.execute(
            "SELECT id, klass_asked, source, klass_answered, correct "
            "FROM attempts WHERE id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            return None
        if row["klass_answered"] is not None:
            return dict(row)   # ya respondido, no re-grabar
        correct = 1 if klass_answered == row["klass_asked"] else 0
        conn.execute(
            "UPDATE attempts SET klass_answered = ?, correct = ? WHERE id = ?",
            (klass_answered, correct, attempt_id),
        )
        conn.commit()
        return {
            "id": row["id"],
            "klass_asked": row["klass_asked"],
            "source": row["source"],
            "klass_answered": klass_answered,
            "correct": correct,
        }


def stats_overall(conn: sqlite3.Connection) -> dict:
    """Devuelve totales, aciertos, accuracy y matriz por clase."""
    with _lock:
        rows = conn.execute(
            "SELECT klass_asked, klass_answered, correct, source "
            "FROM attempts WHERE klass_answered IS NOT NULL"
        ).fetchall()

    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "per_class": {},
            "current_streak": 0,
            "best_streak": 0,
        }

    correct = sum(int(r["correct"]) for r in rows)
    per_class: dict[str, dict[str, int]] = {}
    for r in rows:
        bucket = per_class.setdefault(
            r["klass_asked"], {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(r["correct"])

    # rachas: pasamos por las filas en orden cronologico (autoincrement id ya
    # esta en ese orden) y contamos la racha actual + la mejor historica.
    current = 0
    best = 0
    for r in rows:
        if r["correct"]:
            current += 1
            if current > best:
                best = current
        else:
            current = 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "per_class": per_class,
        "current_streak": current,
        "best_streak": best,
    }


def purge_stale_pending(conn: sqlite3.Connection,
                        ttl_hours: int = PENDING_TTL_HOURS) -> int:
    """Borra filas con respuesta nula mas viejas que ttl_hours.

    Asi la DB no acumula intentos abandonados (pestaña cerrada antes de
    contestar). Devuelve la cantidad de filas borradas.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
              ).isoformat(timespec="seconds")
    with _lock:
        cur = conn.execute(
            "DELETE FROM attempts WHERE klass_answered IS NULL AND ts < ?",
            (cutoff,),
        )
        conn.commit()
        return int(cur.rowcount)
