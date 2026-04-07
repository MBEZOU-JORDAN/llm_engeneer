"""
eval.py — Fonctions d'évaluation RAG
- evaluate_retrieval: MRR, nDCG, keyword coverage
- evaluate_answer: LLM-as-a-judge via Groq (accuracy, completeness, relevance)
"""

import math
import os
from dataclasses import dataclass
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

from implementation.answer import answer_question, retrieve_context
from .test import TestQuestion

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_MODEL_JUDGE = "llama-3.3-70b-versatile"  # Modèle juge (plus puissant)
TOP_K = 5


# ─── RÉSULTATS ────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Résultat de l'évaluation de la récupération"""
    mrr: float               # Mean Reciprocal Rank [0, 1]
    ndcg: float              # Normalized Discounted Cumulative Gain [0, 1]
    keywords_coverage: float # % des keywords trouvés dans les chunks [0, 1]
    keywords_found: int      # Nombre de keywords trouvés
    keywords_total: int      # Nombre total de keywords


@dataclass
class AnswerScore:
    """Score LLM-as-a-judge pour une réponse"""
    accuracy: float = Field(validation_alias="exactitude")
    completeness: float = Field(validation_alias="complétude")
    relevance: float = Field(validation_alias="pertinence")
    overall: float

# ─── MÉTRIQUES DE RETRIEVAL ───────────────────────────────────────────────────

def _compute_mrr(keyword: str, chunk_texts: list[str]) -> float:
    """
    Mean Reciprocal Rank pour un keyword donné.
    Retourne 1/rank si le keyword est trouvé, 0 sinon.
    - Position 1 → score 1.0
    - Position 2 → score 0.5
    - Position 3 → score 0.33
    """
    keyword_lower = keyword.lower()
    for rank, text in enumerate(chunk_texts, start=1):
        if keyword_lower in text.lower():
            return 1.0 / rank
    return 0.0


def _compute_ndcg(keyword: str, chunk_texts: list[str]) -> float:
    """
    Normalized Discounted Cumulative Gain pour un keyword.
    Pénalise les bonnes réponses trouvées tard dans la liste.
    """
    # Relevances binaires: 1 si le keyword est dans le chunk
    relevances = [1 if keyword.lower() in text.lower() else 0 for text in chunk_texts]
    
    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    
    # IDCG (DCG idéal: tous les chunks pertinents en premier)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, top_k: int = TOP_K) -> RetrievalResult:
    """
    Évalue la qualité de la récupération pour une question de test.
    
    Pour chaque keyword:
    - Cherche dans quel chunk il apparaît (et à quel rang)
    - Calcule MRR et nDCG
    Retourne la moyenne sur tous les keywords.
    """
    _, chunk_texts = retrieve_context(test.question, top_k=top_k)
    
    mrr_scores = []
    ndcg_scores = []
    keywords_found = 0
    
    for keyword in test.keywords:
        mrr = _compute_mrr(keyword, chunk_texts)
        ndcg = _compute_ndcg(keyword, chunk_texts)
        
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        
        # Un keyword est "trouvé" s'il apparaît dans au moins un chunk
        if any(keyword.lower() in text.lower() for text in chunk_texts):
            keywords_found += 1
    
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    coverage = keywords_found / len(test.keywords) if test.keywords else 0.0
    
    return RetrievalResult(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_coverage=coverage,
        keywords_found=keywords_found,
        keywords_total=len(test.keywords),
    )


# ─── LLM-AS-A-JUDGE ───────────────────────────────────────────────────────────

class EvaluationScores(BaseModel):
    """Structure JSON retournée par le juge LLM"""
    accuracy: int       # 1-5
    completeness: int   # 1-5
    relevance: int      # 1-5
    reasoning: str      # Explication courte


def evaluate_answer(test: TestQuestion) -> AnswerScore:
    """
    Évalue la qualité de la réponse RAG via LLM-as-a-judge.
    
    Étapes:
    1. Génère la réponse RAG pour la question
    2. Demande au LLM juge de noter accuracy/completeness/relevance
    3. Retourne les scores
    """
    # 1. Générer la réponse RAG
    generated_answer = answer_question(test.question)
    
    # 2. Prompt du juge
    judge_prompt = f"""Tu es un évaluateur expert. Évalue la réponse générée par rapport à la réponse de référence.

QUESTION: {test.question}

RÉPONSE GÉNÉRÉE: {generated_answer}

RÉPONSE DE RÉFÉRENCE: {test.reference_answer}

Évalue la réponse générée sur ces 3 dimensions (score de 1 à 5):
- accuracy (exactitude): Les faits sont-ils corrects ? (5 = parfaitement exact)
- completeness (complétude): La réponse couvre-t-elle tous les points importants ? (5 = complet)
- relevance (pertinence): La réponse est-elle bien ciblée sur la question ? (5 = parfaitement pertinent)

IMPORTANT: Donne uniquement un 5/5 pour une réponse parfaite. Sois strict.

Réponds UNIQUEMENT avec ce JSON (sans markdown, sans explication hors JSON):
{{"accuracy": <1-5>, "completeness": <1-5>, "relevance": <1-5>, "reasoning": "<explication courte>"}}"""
    
    # 3. Appel au juge LLM
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_JUDGE,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Parser le JSON (nettoie les backticks éventuels)
        import json, re
        raw = re.sub(r"```json|```", "", raw).strip()
        scores_data = json.loads(raw)
        scores = EvaluationScores(**scores_data)
        
        overall = (scores.accuracy + scores.completeness + scores.relevance) / 3.0
        
        return AnswerScore(
            accuracy=float(scores.accuracy),
            completeness=float(scores.completeness),
            relevance=float(scores.relevance),
            overall=overall,
        )
    
    except Exception as e:
        print(f"⚠️  Erreur juge LLM: {e}")
        return AnswerScore(accuracy=0, completeness=0, relevance=0, overall=0)


# ─── TEST RAPIDE ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from test import load_tests
    
    tests = load_tests()
    print(f"Test sur la première question...")
    
    t = tests[0]
    print(f"\nQuestion: {t.question}")
    
    ret = evaluate_retrieval(t)
    print(f"\n📊 Retrieval:")
    print(f"   MRR     : {ret.mrr:.4f}")
    print(f"   nDCG    : {ret.ndcg:.4f}")
    print(f"   Coverage: {ret.keywords_coverage:.1%} ({ret.keywords_found}/{ret.keywords_total} keywords)")
    
    ans = evaluate_answer(t)
    print(f"\n🤖 Answer (LLM Judge):")
    print(f"   Accuracy    : {ans.accuracy}/5")
    print(f"   Completeness: {ans.completeness}/5")
    print(f"   Relevance   : {ans.relevance}/5")
    print(f"   Overall     : {ans.overall:.2f}/5")
