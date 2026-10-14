"""Small SQLite persistence layer for the TumorGANet demo backend.

This keeps patient, scan, report, and session data out of the React component.
It is suitable for local development; production deployments should replace the
default SQLite file with a managed database and a real identity provider.
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tumorganet.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        salt_hex, digest_hex = encoded.split("$", 1)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 120_000)
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def init_db() -> None:
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT UNIQUE,
                display_name TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'doctor',
                reset_token TEXT,
                reset_expires_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                history TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT,
                confidence REAL,
                probabilities TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            );
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                scan_id INTEGER,
                doctor_name TEXT NOT NULL,
                decision TEXT NOT NULL,
                notes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES patients(id),
                FOREIGN KEY(scan_id) REFERENCES scans(id)
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT,
                details TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
            );
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "email" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        if "display_name" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
        if "reset_token" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
        if "reset_expires_at" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN reset_expires_at TEXT")
        if conn.execute("SELECT 1 FROM users WHERE username = 'admin'").fetchone() is None:
            password = os.getenv("TUMORGANET_ADMIN_PASSWORD", "change-me")
            conn.execute(
                "INSERT INTO users(username, email, display_name, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("admin", os.getenv("TUMORGANET_ADMIN_EMAIL", "admin@example.com"), "Administrator", hash_password(password), "admin", now()),
            )
        else:
            # Upgrade databases created before email support was added.
            conn.execute(
                "UPDATE users SET email = COALESCE(NULLIF(email, ''), ?), display_name = COALESCE(NULLIF(display_name, ''), ?) WHERE username = 'admin'",
                (os.getenv("TUMORGANET_ADMIN_EMAIL", "admin@example.com"), "Administrator"),
            )


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, username.lower())).fetchone()
    if row and verify_password(password, row["password_hash"]):
        return {"id": row["id"], "username": row["username"], "email": row["email"], "display_name": row["display_name"], "role": row["role"]}
    return None


def create_session(user: Dict[str, Any], days: int = 7) -> str:
    token = secrets.token_urlsafe(48)
    expires_at = datetime.fromtimestamp(datetime.now().timestamp() + days * 86400, timezone.utc).isoformat()
    with connection() as conn:
        conn.execute(
            "INSERT INTO sessions(token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token, user["id"], expires_at, now()),
        )
    return token


def get_session(token: str) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        row = conn.execute(
            "SELECT s.token, s.expires_at, u.id, u.username, u.email, u.display_name, u.role "
            "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?",
            (token,),
        ).fetchone()
        if not row:
            return None
        if row["expires_at"] < now():
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
        return {"id": row["id"], "username": row["username"], "email": row["email"], "display_name": row["display_name"], "role": row["role"]}


def delete_session(token: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def create_audit_event(user_id: Optional[int], action: str, entity_type: str, entity_id: Optional[str] = None, details: str = "") -> Dict[str, Any]:
    record = {"user_id": user_id, "action": action[:80], "entity_type": entity_type[:80], "entity_id": entity_id, "details": details[:2000], "created_at": now()}
    with connection() as conn:
        cursor = conn.execute(
            "INSERT INTO audit_events(user_id, action, entity_type, entity_id, details, created_at) VALUES (:user_id, :action, :entity_type, :entity_id, :details, :created_at)",
            record,
        )
        record["id"] = cursor.lastrowid
    return record


def list_audit_events(limit: int = 100) -> list[Dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT a.*, u.display_name, u.email FROM audit_events a LEFT JOIN users u ON u.id = a.user_id ORDER BY a.created_at DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        ).fetchall()
    return [dict(row) for row in rows]


def create_user(name: str, email: str, password: str) -> Dict[str, Any]:
    email = email.strip().lower()
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters")
    if "@" not in email:
        raise ValueError("A valid email address is required")
    with connection() as conn:
        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise ValueError("An account with this email already exists")
        username = email
        conn.execute(
            "INSERT INTO users(username, email, display_name, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (username, email, name.strip()[:120] or email, hash_password(password), "doctor", now()),
        )
    return authenticate(email, password)


def create_reset_token(email: str) -> Optional[str]:
    token = secrets.token_urlsafe(32)
    expires = datetime.fromtimestamp(datetime.now().timestamp() + 3600, timezone.utc).isoformat()
    with connection() as conn:
        cursor = conn.execute("UPDATE users SET reset_token = ?, reset_expires_at = ? WHERE email = ?", (token, expires, email.strip().lower()))
        return token if cursor.rowcount else None


def reset_password(token: str, password: str) -> bool:
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters")
    with connection() as conn:
        row = conn.execute("SELECT id, reset_expires_at FROM users WHERE reset_token = ?", (token,)).fetchone()
        if not row or not row["reset_expires_at"] or row["reset_expires_at"] < now():
            return False
        conn.execute("UPDATE users SET password_hash = ?, reset_token = NULL, reset_expires_at = NULL WHERE id = ?", (hash_password(password), row["id"]))
    return True


def create_patient(payload: Dict[str, Any]) -> Dict[str, Any]:
    patient_id = payload.get("id") or f"PT-{secrets.randbelow(9000) + 1000}"
    record = {
        "id": patient_id,
        "name": str(payload.get("name", "Unnamed patient")).strip()[:120],
        "age": payload.get("age"),
        "gender": payload.get("gender", "Not specified"),
        "history": payload.get("history", ""),
        "created_at": now(),
    }
    with connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO patients(id, name, age, gender, history, created_at) VALUES (:id, :name, :age, :gender, :history, :created_at)",
            record,
        )
    return record


def list_patients(search: str = "") -> list[Dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM patients WHERE name LIKE ? OR id LIKE ? ORDER BY created_at DESC",
            (f"%{search}%", f"%{search}%"),
        ).fetchall()
    return [dict(row) for row in rows]


def create_scan(payload: Dict[str, Any]) -> Dict[str, Any]:
    patient = create_patient(payload.get("patient", {}))
    record = {
        "patient_id": patient["id"],
        "filename": payload.get("filename", "uploaded-scan"),
        "prediction": payload.get("prediction"),
        "confidence": payload.get("confidence"),
        "probabilities": payload.get("probabilities", "{}"),
        "metadata": payload.get("metadata", "{}"),
        "created_at": now(),
    }
    with connection() as conn:
        cursor = conn.execute(
            "INSERT INTO scans(patient_id, filename, prediction, confidence, probabilities, metadata, created_at) VALUES (:patient_id, :filename, :prediction, :confidence, :probabilities, :metadata, :created_at)",
            record,
        )
        record["id"] = cursor.lastrowid
    return record


def list_scans(patient_id: Optional[str] = None) -> list[Dict[str, Any]]:
    query = "SELECT * FROM scans"
    params: tuple[Any, ...] = ()
    if patient_id:
        query += " WHERE patient_id = ?"
        params = (patient_id,)
    query += " ORDER BY created_at DESC LIMIT 100"
    with connection() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def create_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    patient = create_patient(payload.get("patient", {}))
    record = {
        "patient_id": patient["id"],
        "scan_id": payload.get("scan_id"),
        "doctor_name": str(payload.get("doctor_name", "Unknown doctor"))[:160],
        "decision": str(payload.get("decision", "concur"))[:40],
        "notes": str(payload.get("notes", ""))[:4000],
        "created_at": now(),
    }
    with connection() as conn:
        cursor = conn.execute(
            "INSERT INTO reports(patient_id, scan_id, doctor_name, decision, notes, created_at) VALUES (:patient_id, :scan_id, :doctor_name, :decision, :notes, :created_at)",
            record,
        )
        record["id"] = cursor.lastrowid
    return record


def list_reports(patient_id: Optional[str] = None) -> list[Dict[str, Any]]:
    query = "SELECT * FROM reports"
    params: tuple[Any, ...] = ()
    if patient_id:
        query += " WHERE patient_id = ?"
        params = (patient_id,)
    query += " ORDER BY created_at DESC LIMIT 100"
    with connection() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


init_db()
