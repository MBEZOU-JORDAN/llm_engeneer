"""
test.py — Modèles Pydantic pour les données de test + chargeur
Format: tests.jsonl (une question par ligne)
"""

import os
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Optional 


# ─── MODÈLE PYDANTIC ──────────────────────────────────────────────────────────

class TestQuestion(BaseModel):
    """
    Représente une question de test pour l'évaluation RAG.
    
    Champs:
    - question: La question posée au système RAG
    - keywords: Mots-clés qui DOIVENT apparaître dans les chunks récupérés
    - reference_answer: La réponse de référence (gold standard)
    - category: Catégorie de la question (pour l'analyse par type)
    """
    question: str
    keywords: list[str]
    reference_answer: str
    category: str


# ─── CHARGEUR ─────────────────────────────────────────────────────────────────

# 1. On récupère le dossier où se trouve le fichier test.py lui-même
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. On construit le chemin vers tests.jsonl en partant de ce dossier
TEST_FILE_PATH = os.path.join(CURRENT_DIR, "tests.jsonl")
def load_tests():
    if not os.path.exists(TEST_FILE_PATH):
        raise FileNotFoundError(
            f"Fichier de tests introuvable à l'emplacement : {TEST_FILE_PATH}"
        )
    # ... reste de ton code qui utilise TEST_FILE_PATH ...

def load_tests(filepath: str = TEST_FILE_PATH) -> list[TestQuestion]:
    """
    Charge les questions de test depuis un fichier JSONL.
    Chaque ligne = un JSON object représentant une TestQuestion.
    """
    tests = []
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de tests '{filepath}' introuvable.\n"
            f"Crée-le avec generate_tests.py ou manuellement."
        )
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                tests.append(TestQuestion(**data))
            except json.JSONDecodeError as e:
                print(f"⚠️  Ligne {line_num} ignorée (JSON invalide): {e}")
            except Exception as e:
                print(f"⚠️  Ligne {line_num} ignorée: {e}")
    
    return tests


def get_categories(tests: list[TestQuestion]) -> dict[str, int]:
    """Retourne le nombre de questions par catégorie"""
    from collections import Counter
    return dict(Counter(t.category for t in tests))


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = load_tests()
    print(f"✅ {len(tests)} questions chargées")
    print(f"📊 Catégories: {get_categories(tests)}")
    print(f"\nExemple:")
    if tests:
        t = tests[0]
        print(f"  Question  : {t.question}")
        print(f"  Keywords  : {t.keywords}")
        print(f"  Référence : {t.reference_answer[:80]}...")
        print(f"  Catégorie : {t.category}")
