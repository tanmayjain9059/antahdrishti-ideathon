import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "inspections.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bridge_id TEXT NOT NULL,
            bridge_name TEXT NOT NULL,
            location TEXT NOT NULL,
            image_name TEXT NOT NULL,
            defect_count INTEGER NOT NULL,
            detections TEXT NOT NULL,
            average_confidence REAL NOT NULL,
            health_score INTEGER NOT NULL,
            priority TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            inspection_time TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS engineer_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER NOT NULL,
            engineer_decision TEXT NOT NULL,
            corrected_defect TEXT,
            final_priority TEXT,
            engineer_remarks TEXT,
            feedback_time TEXT NOT NULL,
            FOREIGN KEY (inspection_id) REFERENCES inspections(id)
        )
    """)

    connection.commit()
    connection.close()


def save_inspection(
    bridge_id,
    bridge_name,
    location,
    image_name,
    defect_count,
    detections,
    average_confidence,
    health_score,
    priority,
    recommendation,
    inspection_time
):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO inspections (
            bridge_id,
            bridge_name,
            location,
            image_name,
            defect_count,
            detections,
            average_confidence,
            health_score,
            priority,
            recommendation,
            inspection_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bridge_id,
        bridge_name,
        location,
        image_name,
        defect_count,
        detections,
        average_confidence,
        health_score,
        priority,
        recommendation,
        inspection_time
    ))

    inspection_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return inspection_id


def get_inspections():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            i.id,
            i.bridge_id,
            i.bridge_name,
            i.location,
            i.image_name,
            i.defect_count,
            i.detections,
            i.average_confidence,
            i.health_score,
            i.priority,
            i.recommendation,
            i.inspection_time,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM engineer_feedback f
                    WHERE f.inspection_id = i.id
                )
                THEN 1
                ELSE 0
            END AS has_feedback
        FROM inspections i
        ORDER BY i.id DESC
    """)

    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]


def get_health_history(bridge_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            inspection_time,
            health_score,
            priority
        FROM inspections
        WHERE bridge_id = ?
        ORDER BY id ASC
    """, (bridge_id,))

    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]


def save_feedback(
    inspection_id,
    engineer_decision,
    corrected_defect=None,
    final_priority=None,
    engineer_remarks=None,
    feedback_time=None
):
    if feedback_time is None:
        feedback_time = datetime.now().strftime("%d-%m-%Y %H:%M")

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO engineer_feedback (
            inspection_id,
            engineer_decision,
            corrected_defect,
            final_priority,
            engineer_remarks,
            feedback_time
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        inspection_id,
        engineer_decision,
        corrected_defect,
        final_priority,
        engineer_remarks,
        feedback_time
    ))

    feedback_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return feedback_id


def get_feedback(inspection_id=None):
    connection = get_connection()
    cursor = connection.cursor()

    if inspection_id is None:
        cursor.execute("""
            SELECT
                f.id,
                f.inspection_id,
                f.engineer_decision,
                f.corrected_defect,
                f.final_priority,
                f.engineer_remarks,
                f.feedback_time,
                i.bridge_id,
                i.bridge_name
            FROM engineer_feedback f
            JOIN inspections i
                ON i.id = f.inspection_id
            ORDER BY f.id DESC
        """)
    else:
        cursor.execute("""
            SELECT
                f.id,
                f.inspection_id,
                f.engineer_decision,
                f.corrected_defect,
                f.final_priority,
                f.engineer_remarks,
                f.feedback_time,
                i.bridge_id,
                i.bridge_name
            FROM engineer_feedback f
            JOIN inspections i
                ON i.id = f.inspection_id
            WHERE f.inspection_id = ?
            ORDER BY f.id DESC
        """, (inspection_id,))

    rows = cursor.fetchall()
    connection.close()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    create_database()
    print("Database created successfully.")