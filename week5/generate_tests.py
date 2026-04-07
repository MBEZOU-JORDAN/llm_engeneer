"""
generate_tests.py — Génère des questions de test automatiquement depuis tes documents
Utilise Groq pour créer un tests.jsonl personnalisé à ton corpus
Usage: python generate_tests.py
"""

import json
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. On récupère le dossier où se trouve le fichier test.py lui-même
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_DIR = CURRENT_DIR + "/knowledge_base"
OUTPUT_FILE = f"{CURRENT_DIR}/tests.jsonl"
GROQ_MODEL = "llama-3.3-70b-versatile"
QUESTIONS_PER_CHUNK = 2  # Questions générées par chunk de document

CATEGORIES = ["direct_fact", "temporal", "numerical", "comparative", "relationship", "holistic", "spanning"]

GENERATION_PROMPT = """Tu es un expert en création de données de test pour les systèmes RAG.
À partir du texte suivant, génère exactement {n} questions de test avec leurs métadonnées.

TEXTE:
{text}

Pour chaque question, fournis:
- question: La question (en français)
- keywords: 2-4 mots-clés qui DOIVENT apparaître dans le chunk pour répondre (liste de strings)
- reference_answer: La réponse correcte et complète basée sur le texte
- category: Une parmi {categories}

Catégories:
- direct_fact: Fait direct trouvable dans un seul passage
- temporal: Question de durée, date, délai
- numerical: Chiffres, montants, pourcentages
- comparative: Comparaison entre éléments
- relationship: Relations entre personnes/entités
- holistic: Vue d'ensemble sur tout le document
- spanning: Nécessite plusieurs passages pour répondre

Réponds UNIQUEMENT avec un tableau JSON valide, rien d'autre:
[
  {{
    "question": "...",
    "keywords": ["...", "..."],
    "reference_answer": "...",
    "category": "..."
  }},
  ...
]"""


def load_sample_texts(docs_dir: str, max_chunks: int = 10) -> list[str]:
    """Charge quelques chunks représentatifs des documents"""
    documents = []
    docs_path = Path(docs_dir)
    
    for pdf_path in docs_path.glob("**/*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    
    for txt_path in list(docs_path.glob("**/*.txt")) + list(docs_path.glob("**/*.md")):
        loader = TextLoader(str(txt_path), encoding="utf-8")
        documents.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    
    # Sélectionne des chunks distribués uniformément
    if len(chunks) <= max_chunks:
        selected = chunks
    else:
        step = len(chunks) // max_chunks
        selected = chunks[::step][:max_chunks]
    
    return [chunk.page_content for chunk in selected]


def generate_questions_for_chunk(text: str, client: Groq, n: int = QUESTIONS_PER_CHUNK) -> list[dict]:
    """Génère des questions pour un chunk de texte donné"""
    prompt = GENERATION_PROMPT.format(
        n=n,
        text=text[:1500],  # Limite la taille
        categories=", ".join(CATEGORIES)
    )
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Nettoyer et parser le JSON
        import re
        raw = re.sub(r"```json|```", "", raw).strip()
        questions = json.loads(raw)
        return questions if isinstance(questions, list) else []
    
    except Exception as e:
        print(f"  ⚠️  Erreur génération: {e}")
        return []


def main():
    print("=" * 55)
    print("  GÉNÉRATEUR DE TESTS RAG")
    print("=" * 55)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY manquant dans .env")
        return
    
    client = Groq(api_key=api_key)
    
    print(f"📂 Chargement des documents depuis '{DOCS_DIR}'...")
    texts = load_sample_texts(DOCS_DIR, max_chunks=10)
    print(f"✅ {len(texts)} chunks sélectionnés pour la génération")
    
    all_questions = []
    
    for i, text in enumerate(texts, 1):
        print(f"  🔄 Génération chunk {i}/{len(texts)}...")
        questions = generate_questions_for_chunk(text, client)
        all_questions.extend(questions)
        print(f"  ✅ {len(questions)} questions générées")
    
    # Sauvegarder en JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    print(f"\n🎉 {len(all_questions)} questions sauvegardées dans '{OUTPUT_FILE}'")
    print(f"   Lance maintenant: python evaluation.py")


if __name__ == "__main__":
    main()
