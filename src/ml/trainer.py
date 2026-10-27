"""
trainer.py

Trenira i poredi modele za entity matching koristeći MLflow tracking.

Tri eksperimenta
-----------------
EXP 1 — Baseline
    Featurei : samo fuzzy (f_ratio, f_partial, f_token_sort, f_token_set, f_wratio)
    Cilj     : utvrditi koliko sam fuzzy matcher "zna"

EXP 2 — Fuzzy + engineering
    Featurei : fuzzy + strukturni + token overlap + jaro_winkler + candidate_rank
    Cilj     : da li ručno engineerovani featurei pomažu?

EXP 3 — Fuzzy + engineering + embeddings
    Featurei : sve iz EXP 2 + cosine_sim (sentence transformer)
    Cilj     : da li embeddinzi donose mjerljivu korist?
               Ako razlika u F1 < 2%, izbacujemo ih iz produkcije.

Za svaki eksperiment treniraju se tri modela:
    - Logistic Regression  (interpretabilan baseline)
    - Random Forest        (feature importance, otporan na outliere)
    - XGBoost              (tipično best performance na tabularnim podacima)

Pokretanje
----------
    python -m src.ml.trainer                    ← EXP 1 i 2 (bez embeddings)
    python -m src.ml.trainer --all-experiments  ← sva 3 eksperimenta
    python -m src.ml.trainer --experiment 3     ← samo EXP 3
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

try:
    from src.logger import get_logger
except ImportError:
    from src.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Putanje
# ------------------------------------------------------------------ #
DATASET_PATH         = Path("data/dataset.csv")
DATASET_EMBED_PATH   = Path("data/dataset_with_embeddings.csv")
MODELS_DIR           = Path("models")
ARTIFACTS_DIR        = Path("artifacts")
METRICS_SUMMARY_PATH = Path("artifacts/metrics_summary.csv")
MLFLOW_EXPERIMENT    = "name-matching"

# ------------------------------------------------------------------ #
# Definicija eksperimenata
# ------------------------------------------------------------------ #

FUZZY_FEATURES = [
    "f_ratio", "f_partial", "f_token_sort", "f_token_set", "f_wratio",
]

ENGINEERED_FEATURES = [
    "len_diff", "len_ratio", "token_count_diff",
    "jaccard", "first_token_sim", "shared_fuzzy_tokens",
    "jaro_winkler", "candidate_rank",
]

EMBEDDING_FEATURES = ["cosine_sim"]

EXPERIMENTS = {
    1: {
        "name":             "EXP1_fuzzy_only",
        "features":         FUZZY_FEATURES,
        "needs_embeddings": False,
        "description":      "Samo fuzzy featurei — baseline",
    },
    2: {
        "name":             "EXP2_fuzzy_plus_engineering",
        "features":         FUZZY_FEATURES + ENGINEERED_FEATURES,
        "needs_embeddings": False,
        "description":      "Fuzzy + strukturni + token overlap + jaro_winkler",
    },
    3: {
        "name":             "EXP3_all_features",
        "features":         FUZZY_FEATURES + ENGINEERED_FEATURES + EMBEDDING_FEATURES,
        "needs_embeddings": True,
        "description":      "Sve + cosine_sim (sentence transformer)",
    },
}

# ------------------------------------------------------------------ #
# Modeli
# ------------------------------------------------------------------ #

def get_models(pos_weight: float) -> dict:
    """
    Vraća modele za treniranje.

    pos_weight = negativnih / pozitivnih
    Govori modelu koliko je greška na pozitivnom primjeru skuplja.

    Primjer: 500 negativnih / 117 pozitivnih ≈ 4.27
    → Greška na pozitivnom se računa 4.27x teže.

    Logistic Regression: Pipeline sa StandardScalerom jer je
    osjetljiva na skalu featura (f_wratio je 0-1, len_diff može biti 20+).
    RF i XGBoost ne trebaju scaling jer koriste stabla odluke.
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=pos_weight,  # stvarni omjer neg/pos
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


# ------------------------------------------------------------------ #
# Metrike
# ------------------------------------------------------------------ #

def compute_top1_top3(df_full: pd.DataFrame, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Računa top-1 i top-3 accuracy PO GRUPAMA.

    Svaka grupa = jedno PDF ime sa svojim kandidatima.
    Dataset ima 3 reda po PDF imenu (3 kandidata).

    Top-1: Da li je tačan kandidat dobio NAJVIŠI score u grupi?
    Top-3: Da li je tačan kandidat u top 3? (uvijek da ako imamo 3 kandidata,
           ali korisno kad imamo više kandidata ili kad top_n varira)

    Ovo je ispravna implementacija — za razliku od aproksimacije
    top1_acc = precision koja ignoriše grupnu strukturu.

    Primjer:
        Grupa "Peric Pera":
            kandidat "Perić Pero"    label=1  prob=0.94  ← tačan, ima najviši score ✓
            kandidat "Perić Petar"   label=0  prob=0.21
            kandidat "Perić Predrag" label=0  prob=0.08
        → top1 = 1 (pogodak)

        Grupa "Kovac M":
            kandidat "Kovač Milan"   label=1  prob=0.68  ← tačan, ali nije prvi!
            kandidat "Kovač Marko"   label=0  prob=0.71  ← model misli da je ovo match
            kandidat "Kovač Mirko"   label=0  prob=0.12
        → top1 = 0 (promašaj)
    """
    df_eval = df_full[["pdf_name", "label"]].copy()
    df_eval["prob"] = y_prob

    top1_hits = []
    top3_hits = []

    for pdf_name, group in df_eval.groupby("pdf_name"):
        # Sortiraj po score-u opadajuće
        group_sorted = group.sort_values("prob", ascending=False)

        # Tačan kandidat — onaj sa label=1
        has_positive = group_sorted["label"].sum() > 0
        if not has_positive:
            continue  # preskoči grupe bez pozitivnog primjera

        # Top-1: da li je PRVI u sortiranoj listi tačan?
        top1_hits.append(int(group_sorted.iloc[0]["label"] == 1))

        # Top-3: da li je tačan kandidat u prvih 3?
        top3 = group_sorted.head(3)
        top3_hits.append(int(top3["label"].sum() > 0))

    top1_acc = np.mean(top1_hits) if top1_hits else 0.0
    top3_acc = np.mean(top3_hits) if top3_hits else 0.0

    return round(float(top1_acc), 4), round(float(top3_acc), 4)


def compute_metrics(
    df_full: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Računa sve metrike za jedan model/eksperiment.

    Parametri
    ----------
    df_full : DataFrame sa pdf_name kolonom — potreban za grupnu top-1/top-3
    y_true  : stvarne labele (0 ili 1)
    y_pred  : predikcije modela (0 ili 1, na pragu 0.5)
    y_prob  : vjerovatnoće modela za klasu 1 (između 0 i 1)

    Metrike
    -------
    precision  — od svih "match" predikcija, koliko je tačnih?
                 Greška = pogrešan korisnik unesen u program (ozbiljno)

    recall     — od svih stvarnih matcheva, koliko je pronađeno?
                 Greška = propušten unos, ide u unresolved (manje ozbiljno)

    f1         — harmonijska sredina; glavna metrika za poređenje eksperimenata

    roc_auc    — koliko dobro model rangira pozitivne ispred negativnih
                 Nije osjetljiv na nebalans klasa kao accuracy

    top1_acc   — za svako PDF ime: da li je tačan kandidat rangiran PRVI?
                 Ključno za auto-match mod u decision engineu

    top3_acc   — za svako PDF ime: da li je tačan kandidat u top 3?
                 Ključno za "ask user" mod — pokazuješ korisniku 3 opcije
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    top1_acc, top3_acc = compute_top1_top3(df_full, y_prob)

    return {
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall), 4),
        "f1":          round(float(f1), 4),
        "roc_auc":     round(float(roc_auc), 4),
        "top1_acc":    top1_acc,
        "top3_acc":    top3_acc,
        "support_pos": int(y_true.sum()),
        "support_neg": int((1 - y_true).sum()),
    }


# ------------------------------------------------------------------ #
# Grafovi (artifacts)
# ------------------------------------------------------------------ #

def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    exp_name: str,
    output_dir: Path,
) -> Path:
    """
    Confusion matrix — 4 polja koja sve govore:

                    Model: MATCH    Model: NO MATCH
    Stvarno MATCH       TP               FN
    Stvarno NO MATCH    FP               TN

    FP = lažni alarm = pogrešan korisnik unesen u program
    FN = propušteni match = ide u unresolved, korisnik mora ručno
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Match", "Match"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name} — {exp_name}", fontsize=11)
    plt.tight_layout()

    path = output_dir / f"cm_{exp_name}_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _save_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
    exp_name: str,
    output_dir: Path,
) -> Path | None:
    """
    Feature importance graf — koji feature je model smatrao najbitnijim?

    Random Forest: prosječna redukcija Gini impurityja po splitu
    XGBoost: gain — koliko svaki feature smanjuje grešku
    Logistic Regression: apsolutne vrijednosti koeficijenata

    Visok importance za feature X znači:
    "Kad model ne zna šta da radi, gleda X da donese odluku."
    """
    # Izvuci importance iz različitih tipova modela
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            return None
    else:
        return None

    # Sortiraj opadajuće po važnosti
    indices            = np.argsort(importances)
    sorted_features    = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.45)))
    bars = ax.barh(sorted_features, sorted_importances, color="steelblue", height=0.6)

    # Dodaj vrijednosti na barove
    for bar, val in zip(bars, sorted_importances):
        ax.text(
            val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9
        )

    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Feature Importance — {model_name}\n{exp_name}", fontsize=11)
    ax.set_xlim(0, max(sorted_importances) * 1.2)
    plt.tight_layout()

    path = output_dir / f"fi_{exp_name}_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _save_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    exp_name: str,
    output_dir: Path,
) -> Path:
    """
    Precision-Recall kriva — kako se mijenja precision i recall
    kad pomjeraš prag odluke (default je 0.5).

    Čitanje:
    - Svaka tačka na krivoj = jedan prag
    - Gornji desni ugao = idealan model (precision=1, recall=1)
    - Pomicanjem praga ulijevo: recall raste, precision pada
    - Pomicanjem praga udesno: precision raste, recall pada

    Koristiti za:
    - Biranje pragova za decision engine (auto_match, ask_user, skip)
    - Razumijevanje trade-offa između false positive i false negative
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, "b-", linewidth=2, label=model_name)

    # Označi prag 0.5 (default)
    idx_05 = np.argmin(np.abs(thresholds - 0.5)) if len(thresholds) > 0 else 0
    ax.scatter(
        recall_vals[idx_05], precision_vals[idx_05],
        color="red", zorder=5, s=80, label="Prag = 0.5"
    )

    # Označi prag 0.8 (kandidat za auto_match)
    idx_08 = np.argmin(np.abs(thresholds - 0.8)) if len(thresholds) > 0 else 0
    ax.scatter(
        recall_vals[idx_08], precision_vals[idx_08],
        color="green", zorder=5, s=80, marker="^", label="Prag = 0.8"
    )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall kriva\n{model_name} — {exp_name}", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = output_dir / f"pr_{exp_name}_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
# Treniranje jednog modela
# ------------------------------------------------------------------ #

def train_one(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    model,
    model_name: str,
    exp_name: str,
    artifacts_dir: Path,
) -> dict:
    """
    Trenira jedan model koristeći 5-fold stratified cross-validation.

    Zašto cross-validation a ne train/test split?
    Sa ~620 primjera, split bi dao samo ~124 test primjera — premalo
    za pouzdane metrike. Cross-validation koristi SVE primjere za
    testiranje (svaki jednom) i daje stabilniju procjenu.

    Zašto Stratified?
    Osigurava da svaki fold ima isti omjer pozitivnih/negativnih.
    Bez toga, neki fold može slučajno imati 0 pozitivnih primjera.

    Na kraju treniramo finalni model na SVIM podacima —
    cross-validation nam daje samo metrike, ne produkcijski model.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # cross_val_predict: svaki primjer dobija predikciju kad je bio u test foldu
    # Nema leakage — model nikad ne vidi primjer i u treniranju i testiranju
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    metrics = compute_metrics(df, y.values, y_pred, y_prob)

    # Sačuvaj predikcije za error analysis u notebooku
    preds_df = df[["pdf_name", "candidate_name", "filijala", "label", "neg_type"]].copy()
    preds_df["y_pred"] = y_pred
    preds_df["y_prob"] = y_prob.round(4)

    preds_path = artifacts_dir / f"predictions_{exp_name}_{model_name}.csv"
    preds_df.to_csv(preds_path, index=False, encoding="utf-8")

    # Finalni model na svim podacima — za produkcijsko korišćenje
    model.fit(X, y)

    return {
        "model":      model,
        "metrics":    metrics,
        "y_pred":     y_pred,
        "y_prob":     y_prob,
        "y_true":     y.values,
        "preds_path": preds_path,
    }


# ------------------------------------------------------------------ #
# Pokretanje jednog eksperimenta
# ------------------------------------------------------------------ #

def run_experiment(exp_id: int, df: pd.DataFrame) -> list[dict]:
    """
    Pokreće jedan eksperiment — trenira sva 3 modela i loguje u MLflow.
    Vraća listu summary redova za finalni poređeni CSV.
    """
    exp_config   = EXPERIMENTS[exp_id]
    exp_name     = exp_config["name"]
    feature_cols = exp_config["features"]
    description  = exp_config["description"]

    # Provjeri da li su featurei u datasetu
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        logger.error(
            f"[{exp_name}] Featurei nisu u datasetu: {missing}\n"
            f"Pokreni dataset_builder sa odgovarajućim flagovima."
        )
        return []

    X = df[feature_cols]
    y = df["label"]

    # Stvarni omjer negativnih i pozitivnih — za XGBoost scale_pos_weight
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    pos_weight = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0

    logger.info(
        f"\n{'='*60}\n"
        f"Eksperiment {exp_id}: {description}\n"
        f"Featurei ({len(feature_cols)}): {feature_cols}\n"
        f"Dataset: {len(df)} redova | {n_pos} pozitivnih | {n_neg} negativnih\n"
        f"pos_weight za XGBoost: {pos_weight}\n"
        f"{'='*60}"
    )

    artifacts_dir = ARTIFACTS_DIR / exp_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    summary_rows = []

    for model_name, model in get_models(pos_weight).items():
        logger.info(f"  Treniram: {model_name}...")

        with mlflow.start_run(run_name=f"{exp_name}__{model_name}"):

            # --- Params ---
            mlflow.log_params({
                "experiment_id":   exp_id,
                "experiment_name": exp_name,
                "model_type":      model_name,
                "feature_set":     json.dumps(feature_cols),
                "n_features":      len(feature_cols),
                "n_samples":       len(df),
                "n_positive":      n_pos,
                "n_negative":      n_neg,
                "pos_weight":      pos_weight,
                "cv_folds":        5,
                "random_seed":     42,
            })

            # --- Treniranje ---
            result = train_one(df, X, y, model, model_name, exp_name, artifacts_dir)

            # --- Metrics ---
            mlflow.log_metrics(result["metrics"])
            mlflow.log_artifact(str(result["preds_path"]))

            logger.info(
                f"    precision={result['metrics']['precision']:.3f}  "
                f"recall={result['metrics']['recall']:.3f}  "
                f"f1={result['metrics']['f1']:.3f}  "
                f"roc_auc={result['metrics']['roc_auc']:.3f}  "
                f"top1={result['metrics']['top1_acc']:.3f}  "
                f"top3={result['metrics']['top3_acc']:.3f}"
            )

            # --- Artifacts ---
            cm_path = _save_confusion_matrix(
                result["y_true"], result["y_pred"],
                model_name, exp_name, artifacts_dir
            )
            mlflow.log_artifact(str(cm_path))

            fi_path = _save_feature_importance(
                result["model"], feature_cols,
                model_name, exp_name, artifacts_dir
            )
            if fi_path:
                mlflow.log_artifact(str(fi_path))

            pr_path = _save_pr_curve(
                result["y_true"], result["y_prob"],
                model_name, exp_name, artifacts_dir
            )
            mlflow.log_artifact(str(pr_path))

            # Classification report kao tekst
            report      = classification_report(
                result["y_true"], result["y_pred"],
                target_names=["No Match", "Match"]
            )
            report_path = artifacts_dir / f"report_{exp_name}_{model_name}.txt"
            report_path.write_text(report, encoding="utf-8")
            mlflow.log_artifact(str(report_path))

            # --- Sačuvaj model ---
            mlflow.sklearn.log_model(result["model"], artifact_path="model")
            model_path = MODELS_DIR / f"{exp_name}__{model_name}.pkl"
            joblib.dump(result["model"], model_path)

            # Akumuliraj summary red
            summary_rows.append({
                "experiment_id":   exp_id,
                "experiment_name": exp_name,
                "model_type":      model_name,
                **result["metrics"],
            })

    logger.info(f"Eksperiment {exp_id} završen.")
    return summary_rows


# ------------------------------------------------------------------ #
# Heatmapa — poređenje svih eksperimenata i modela
# ------------------------------------------------------------------ #

def _save_f1_heatmap(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    """
    F1 heatmapa — redovi su eksperimenti, kolone su modeli.

    Jednim pogledom vidiš koji eksperiment i koji model daje
    najviši F1. Tamnija boja = bolji F1.

    Čitanje:
    - Ako EXP2 >> EXP1: engineerovani featurei pomažu
    - Ako EXP3 ≈ EXP2:  cosine_sim ne donosi korist, izbaciš ga
    - Ako XGBoost >> RF: stabla sa boostingom bolja od bagging-a
    """
    pivot = summary_df.pivot(
        index="experiment_name",
        columns="model_type",
        values="f1"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGn",
        vmin=0.5,
        vmax=1.0,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "F1 Score"},
    )
    ax.set_title("F1 Score — poređenje eksperimenata i modela", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=15, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = output_dir / "heatmap_f1.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"F1 heatmapa sačuvana: {path}")
    return path


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Trenira modele i loguje u MLflow.")
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2, 3],
        help="Pokreni samo jedan eksperiment"
    )
    parser.add_argument(
        "--all-experiments", action="store_true",
        help="Pokreni sva 3 eksperimenta"
    )
    args = parser.parse_args()

    # Odredi koje eksperimente pokrenuti
    if args.all_experiments:
        exp_ids = [1, 2, 3]
    elif args.experiment:
        exp_ids = [args.experiment]
    else:
        exp_ids = [1, 2]

    # Učitaj odgovarajući dataset
    needs_embeddings = any(EXPERIMENTS[e]["needs_embeddings"] for e in exp_ids)
    dataset_path     = DATASET_EMBED_PATH if needs_embeddings else DATASET_PATH

    if not dataset_path.exists():
        logger.error(
            f"Dataset nije pronađen: {dataset_path}\n"
            f"Pokreni prvo:\n"
            f"  python -m src.ml.dataset_builder"
            + (" --embeddings" if needs_embeddings else "")
        )
        return

    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset učitan: {dataset_path} ({len(df)} redova)")

    # Pokreni eksperimente i sakupi summary redove
    all_summary_rows = []
    for exp_id in exp_ids:
        rows = run_experiment(exp_id, df)
        all_summary_rows.extend(rows)

    # Sačuvaj summary CSV i heatmapu
    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)

        # Ako postoji stari summary (prethodnih eksperimenata), spoji
        if METRICS_SUMMARY_PATH.exists():
            existing    = pd.read_csv(METRICS_SUMMARY_PATH)
            summary_df  = pd.concat([existing, summary_df], ignore_index=True)
            summary_df  = summary_df.drop_duplicates(
                subset=["experiment_name", "model_type"], keep="last"
            )

        METRICS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(METRICS_SUMMARY_PATH, index=False)
        logger.info(f"Summary sačuvan: {METRICS_SUMMARY_PATH}")

        # Heatmapa samo ako imamo bar 2 eksperimenta
        if summary_df["experiment_name"].nunique() >= 2:
            heatmap_path = _save_f1_heatmap(summary_df, ARTIFACTS_DIR)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            with mlflow.start_run(run_name="__heatmap_summary__"):
                mlflow.log_artifact(str(heatmap_path))
                mlflow.log_artifact(str(METRICS_SUMMARY_PATH))

    logger.info(
        "\nSvi eksperimenti završeni!\n"
        "Pokreni MLflow UI:\n"
        "    mlflow ui\n"
        "→  http://localhost:5000"
    )


if __name__ == "__main__":
    main()