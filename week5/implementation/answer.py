"""
answer.py — Fonction de réponse RAG
Récupère le contexte pertinent et génère une réponse via Groq (gratuit)
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modèle LLM via Groq (gratuit, rapide)
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"    # Rapide et gratuit

TOP_K = 5  # Nombre de chunks récupérés

# ─── INITIALISATION (une seule fois, mise en cache) ───────────────────────────
_vectorstore = None
_groq_client = None


def get_vectorstore() -> Chroma:
    """Charge le vector store (singleton)"""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    return _vectorstore


def get_groq_client() -> Groq:
    """Initialise le client Groq (singleton)"""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY manquant dans .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def retrieve_context(question: str, top_k: int = TOP_K) -> tuple[list, list[str]]:
    """
    Récupère les chunks les plus pertinents pour la question.
    Retourne: (documents, texts)
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=top_k)
    texts = [doc.page_content for doc in docs]
    return docs, texts


def answer_question(question: str, top_k: int = TOP_K) -> str:
    """
    Pipeline RAG complet:
    1. Retrieve: récupère les chunks pertinents
    2. Augment: construit le prompt avec le contexte
    3. Generate: génère la réponse via Groq
    """
    # 1. Retrieve
    _, context_texts = retrieve_context(question, top_k=top_k)
    context = "\n\n---\n\n".join(context_texts)
    
    # 2. Augment
    prompt = f"""Tu es un assistant expert. Utilise UNIQUEMENT le contexte fourni pour répondre à la question.
Si la réponse n'est pas dans le contexte, dis clairement que tu ne sais pas.
Ne fais pas de suppositions au-delà du contexte.

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE:"""
    
    # 3. Generate
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    
    return response.choices[0].message.content.strip()


# ─── TEST RAPIDE ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    question = "Quelle est la politique de remboursement des frais de déplacement ?"
    print(f"Question: {question}")
    print(f"Réponse: {answer_question(question)}")
