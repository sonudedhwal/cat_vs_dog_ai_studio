"""MySQL / MariaDB layer (works with XAMPP + phpMyAdmin defaults)."""
import os
import mysql.connector
import pandas as pd

DB_NAME = os.getenv("DB_NAME", "cat_dog_db")


def _config(with_db=True):
    cfg = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
    }
    if with_db:
        cfg["database"] = DB_NAME
    return cfg


def get_conn(with_db=True):
    return mysql.connector.connect(**_config(with_db))


def init_db():
    """Create database + table if missing."""
    conn = get_conn(with_db=False)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4")
    cur.execute(f"USE `{DB_NAME}`")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            label VARCHAR(20) NOT NULL,
            confidence FLOAT NOT NULL,
            cat_prob FLOAT NOT NULL,
            dog_prob FLOAT NOT NULL,
            model_mode VARCHAR(30),
            ai_description TEXT,
            correct_label VARCHAR(20) NULL,
            image_data LONGBLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def save_prediction(filename, result, image_bytes, ai_description=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO predictions
           (filename, label, confidence, cat_prob, dog_prob, model_mode, ai_description, image_data)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (filename, result["label"], result["confidence"], result["cat_prob"],
         result["dog_prob"], result["mode"], ai_description, image_bytes),
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def update_description(pred_id, text):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE predictions SET ai_description=%s WHERE id=%s", (text, pred_id))
    conn.commit()
    cur.close()
    conn.close()


def update_feedback(pred_id, correct_label):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE predictions SET correct_label=%s WHERE id=%s", (correct_label, pred_id))
    conn.commit()
    cur.close()
    conn.close()


def delete_prediction(pred_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE id=%s", (pred_id,))
    conn.commit()
    cur.close()
    conn.close()


def get_history(limit=200) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql(
        """SELECT id, filename, label, confidence, cat_prob, dog_prob, model_mode,
                  correct_label, ai_description, created_at
           FROM predictions ORDER BY id DESC LIMIT %s""",
        conn, params=(limit,),
    )
    conn.close()
    return df


def get_image(pred_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT image_data FROM predictions WHERE id=%s", (pred_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None


def get_stats() -> dict:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), AVG(confidence) FROM predictions")
    total, avg_conf = cur.fetchone()
    cur.execute("SELECT label, COUNT(*) FROM predictions GROUP BY label")
    by_label = dict(cur.fetchall())
    cur.execute(
        "SELECT COUNT(*), SUM(label = correct_label) FROM predictions WHERE correct_label IS NOT NULL"
    )
    reviewed, correct = cur.fetchone()
    cur.execute(
        "SELECT DATE(created_at) d, COUNT(*) FROM predictions GROUP BY d ORDER BY d"
    )
    per_day = cur.fetchall()
    cur.close()
    conn.close()
    return {
        "total": total or 0,
        "avg_conf": float(avg_conf or 0),
        "by_label": by_label,
        "reviewed": reviewed or 0,
        "correct": int(correct or 0),
        "per_day": per_day,
    }
