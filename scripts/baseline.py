"""
baseline.py — koliko dobro radi sam fuzzy matcher, bez ML modela?

Pokretanje (iz root foldera projekta):
    python scripts/baseline.py

Ništa ne upisuje — bazu otvara samo za čitanje.
"""
import sys
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DB_PATH
from src.fuzzy_matcher import get_candidates, normalize_name

TOP_N = 5  # isto kao u automation_app.py


# ------------------------------------------------------------------ #
# DIO 1 — Ista data kao model (dataset.csv)
# Rang 1 u datasetu = kandidat sa najvećim WRatio = fuzzy baseline.
# Poredi se direktno sa top1_acc iz metrics_summary.csv.
# ------------------------------------------------------------------ #
print("=" * 60)
print("DIO 1 — Fuzzy baseline vs. model (isti dataset)")
print("=" * 60)

df = pd.read_csv(ROOT / "data" / "dataset.csv")
rank1 = df[df["candidate_rank"] == 1].drop_duplicates(subset="pdf_name")

n_groups = df["pdf_name"].nunique()
n_hit = int(rank1["label"].sum())
baseline_top1 = n_hit / n_groups

print(f"Broj PDF imena (grupa):        {n_groups}")
print(f"Fuzzy rang 1 je tačan:         {n_hit}/{n_groups}  ({baseline_top1:.1%})")
print(f"Fuzzy rang 1 je pogrešan:      {n_groups - n_hit}")

summary_path = ROOT / "artifacts" / "metrics_summary.csv"
if summary_path.exists():
    summary = pd.read_csv(summary_path)
    print("\nModel top-1 (cross-validation), za poređenje:")
    for _, r in summary.iterrows():
        hits = round(r["top1_acc"] * n_groups)
        print(f"  {r['experiment_name']:<45} {r['model_type']:<20} "
              f"{r['top1_acc']:.1%}  (~{hits}/{n_groups})")

# Gdje fuzzy griješi — da vidiš kakve su to greške
wrong = rank1[rank1["label"] == 0][["pdf_name", "candidate_name"]]
if not wrong.empty:
    print("\nImena gdje fuzzy rang 1 NIJE tačan:")
    for _, r in wrong.iterrows():
        correct = df[(df["pdf_name"] == r["pdf_name"]) & (df["label"] == 1)]["candidate_name"]
        correct = correct.iloc[0] if not correct.empty else "?"
        print(f"  PDF: {r['pdf_name']:<30} fuzzy: {r['candidate_name']:<30} tačno: {correct}")


# ------------------------------------------------------------------ #
# DIO 2 — Realni scenario (kao u aplikaciji)
# Aliasi koje je potvrdio čovjek, bez ml_auto.
# Na kojoj poziciji fuzzy vraća tačno ime?
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print(f"DIO 2 — Pozicija tačnog imena u fuzzy top {TOP_N} (potvrdio čovjek)")
print("=" * 60)

conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
aliases = pd.read_sql(
    "SELECT pdf_name, web_name, source FROM name_aliases "
    "WHERE source IN ('manual', 'ml_confirmed')",
    conn,
)
conn.close()

positions = []
for _, a in aliases.iterrows():
    cands = get_candidates(a["pdf_name"], filijala=None, top_n=TOP_N)
    names = [normalize_name(c["name"]) for c in cands]
    target = normalize_name(a["web_name"])
    pos = names.index(target) + 1 if target in names else None
    positions.append(pos)

aliases["position"] = positions
n = len(aliases)

def share(mask):
    k = int(mask.sum())
    return f"{k}/{n}  ({k / n:.1%})"

print(f"Ukupno aliasa:                 {n}")
print(f"Tačno ime na poziciji 1:       {share(aliases['position'] == 1)}")
print(f"Tačno ime u top 3:             {share(aliases['position'].between(1, 3))}")
print(f"Tačno ime u top {TOP_N}:             {share(aliases['position'].between(1, TOP_N))}")
print(f"Nije u top {TOP_N} uopšte:           {share(aliases['position'].isna())}")

print("\nPo izvoru:")
for src, g in aliases.groupby("source"):
    p1 = (g["position"] == 1).mean()
    print(f"  {src:<14} n={len(g):<4} pozicija 1: {p1:.1%}")

missing = aliases[aliases["position"].isna()]
if not missing.empty:
    print(f"\nImena koja fuzzy NE nalazi u top {TOP_N} (ni model ih ne može riješiti):")
    for _, r in missing.iterrows():
        print(f"  PDF: {r['pdf_name']:<30} tačno: {r['web_name']}")