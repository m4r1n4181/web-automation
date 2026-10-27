import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from src.database import Database


def load_customers_from_csv(csv_path: Path):
    """
    Zamjenjuje cijelu customers tabelu sadrzajem CSV-a.
    Aliasi (name_aliases) se ne diraju.
    """
    db = Database()
    df = pd.read_csv(csv_path).fillna("")

    with db.connection() as conn:
        old_count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        print(f"Trenutno u bazi: {old_count} | u novom CSV-u: {len(df)}")

        # Zastita: ako je novi scrape manji od postojeceg, nesto nije u redu
        if len(df) < old_count:
            print("Novi CSV ima MANJE kupaca od baze - prekidam, nista nije promijenjeno.")
            return

        conn.execute("DELETE FROM customers")
        for _, row in df.iterrows():
            conn.execute("""
                INSERT INTO customers (customer_id, name, municipality, place_norm, filijala)
                VALUES (?, ?, ?, ?, ?)
            """, (
                row.get("id", ""),
                row.get("name", ""),
                row.get("municipality", ""),
                row.get("place_norm", ""),
                row.get("filijala", ""),
            ))

    print(f"Upisano {len(df)} kupaca (bilo {old_count}, novih {len(df) - old_count}).")


if __name__ == "__main__":
    load_customers_from_csv(SCRIPTS_DIR / "customers_with_filijala.csv")
