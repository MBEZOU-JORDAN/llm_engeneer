"""
ingest.py — Charge les documents et les indexe dans ChromaDB
Supporte: PDF, TXT, MD
Modèles d'embedding: HuggingFace (gratuit) ou OpenAI
"""

import os
import sys
import glob
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DOCS_DIR = "week5/knowledge-base"          # Dossier contenant tes documents
CHROMA_DIR = "chroma_db"             # Dossier de stockage du vector store
COLLECTION_NAME = "rag_collection"

# Paramètres de chunking (à faire varier pour l'évaluation)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Modèle d'embedding (gratuit, ~500MB téléchargé une fois)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative plus puissante (plus lent au premier run):
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ─── FONCTIONS ────────────────────────────────────────────────────────────────

def load_documents(docs_dir: str) -> list:
    """Charge tous les documents du dossier knowledge_base"""
    documents = []
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        print(f"⚠️  Dossier '{docs_dir}' introuvable. Création...")
        docs_path.mkdir(parents=True)
        print(f"📁 Place tes fichiers PDF/TXT dans '{docs_dir}/' puis relance ingest.py")
        sys.exit(0)
    
    # PDF
    for pdf_path in docs_path.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ PDF chargé: {pdf_path.name} ({len(docs)} pages)")
        except Exception as e:
            print(f"❌ Erreur {pdf_path.name}: {e}")
    
    # TXT
    for txt_path in docs_path.glob("**/*.txt"):
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ TXT chargé: {txt_path.name} ({len(docs)} doc)")
        except Exception as e:
            print(f"❌ Erreur {txt_path.name}: {e}")
    
    # MD
    for md_path in docs_path.glob("**/*.md"):
        try:
            loader = TextLoader(str(md_path), encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ MD chargé: {md_path.name} ({len(docs)} doc)")
        except Exception as e:
            print(f"❌ Erreur {md_path.name}: {e}")
    
    if not documents:
        print(f"⚠️  Aucun document trouvé dans '{docs_dir}/'")
        print("   Supporte: .pdf, .txt, .md")
        sys.exit(0)
    
    return documents


def chunk_documents(documents: list) -> list:
    """Découpe les documents en chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"\n📦 Chunking: {len(documents)} docs → {len(chunks)} chunks")
    print(f"   Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    """Crée ou recrée le vector store ChromaDB"""
    import shutil
    
    # Supprimer l'ancien vector store pour repartir propre
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"🗑️  Ancien vector store supprimé")
    
    print(f"🔄 Chargement du modèle d'embedding: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print(f"⚡ Création du vector store ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist()
    
    count = vectorstore._collection.count()
    print(f"✅ Vector store créé: {count} vecteurs indexés")
    return vectorstore


def main():
    print("=" * 55)
    print("  RAG INGEST — Indexation des documents")
    print("=" * 55)
    
    documents = load_documents(DOCS_DIR)
    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)
    
    print("\n🎉 Ingestion terminée! Prêt pour l'évaluation.")
    print(f"   Vectorstore: {CHROMA_DIR}/")
    print(f"   Collection : {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
