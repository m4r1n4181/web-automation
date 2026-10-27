import sqlite3
import sys
sys.path.insert(0, '.')
from src.database import Database

# Putanja do stare baze iz koje se uvoze aliasi:
#   python scripts/import_aliases.py putanja/do/stare/payments.db
if len(sys.argv) < 2:
    sys.exit("Upotreba: python scripts/import_aliases.py <putanja do stare payments.db>")
STARA_BAZA = sys.argv[1]

old_conn = sqlite3.connect(STARA_BAZA)
old_conn.row_factory = sqlite3.Row
rows = old_conn.execute("SELECT * FROM name_aliases").fetchall()
old_conn.close()

db = Database()
imported = 0
for row in rows:
    db.save_alias(
        pdf_name   = row["pdf_name"],
        web_name   = row["web_name"],
        confidence = row["confidence"],
        confirmed  = bool(row["confirmed"]),
        source     = row["source"],
    )
    imported += 1

print(f"Importovano {imported} aliasa.")
print(db.get_stats())