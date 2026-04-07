"""
app.py — Dashboard Streamlit pour l'évaluation RAG
Interface moderne, scalable et interactive.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import time

from evaluation.test import load_tests
from evaluation.eval import evaluate_retrieval, evaluate_answer
from implementation.answer import answer_question

# ─── CONFIGURATION DE LA PAGE ───
st.set_page_config(
    page_title="RAG Evaluation Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CHARGEMENT EN CACHE ───
@st.cache_data
def get_test_data():
    """Mise en cache des tests pour éviter de relire le fichier JSONL en boucle"""
    try:
        return load_tests()
    except Exception as e:
        st.error(f"Erreur lors du chargement des tests : {e}")
        return []

tests = get_test_data()

# ─── SIDEBAR ───
with st.sidebar:
    st.title("⚙️ Paramètres")
    st.info(f"**{len(tests)}** questions de test chargées.")
    st.markdown("---")
    st.markdown("🛠️ **Modèles utilisés :**")
    st.markdown("- **Embedding:** `all-MiniLM-L6-v2`")
    st.markdown("- **RAG LLM:** `llama-4-scout-17b-16e-instruct`")
    st.markdown("- **Judge LLM:** `llama-3.3-70b-versatile`")

st.title("🧠 RAG Evaluation Studio")
st.markdown("Évalue les performances de ton système Retrieval-Augmented Generation en un clic.")

# ─── TABS ───
tab1, tab2, tab3 = st.tabs(["📊 Évaluation Retrieval", "⚖️ Évaluation Réponse (Judge)", "💬 Chat Interactif"])

# ─── TAB 1 : RETRIEVAL ───
with tab1:
    st.header("Évaluation de la Récupération (Retrieval)")
    st.write("Mesure la capacité du Vector Store (Chroma) à ramener les bons chunks.")
    
    if st.button("▶ Lancer l'évaluation Retrieval", type="primary"):
        progress_text = "Évaluation en cours..."
        my_bar = st.progress(0, text=progress_text)
        
        results = []
        for i, test in enumerate(tests):
            res = evaluate_retrieval(test)
            results.append({
                "Catégorie": test.category,
                "Question": test.question[:50] + "...",
                "MRR": res.mrr,
                "NDCG": res.ndcg,
                "Coverage": res.keywords_coverage,
            })
            my_bar.progress((i + 1) / len(tests), text=f"Traitement de la question {i+1}/{len(tests)}")
            
        df_retrieval = pd.DataFrame(results)
        
        # Affichage des KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("MRR Moyen", f"{df_retrieval['MRR'].mean():.2f}")
        col2.metric("nDCG Moyen", f"{df_retrieval['nDCG'].mean():.2f}")
        col3.metric("Coverage Moyen", f"{df_retrieval['Coverage'].mean():.1%}")
        
        # Graphique
        st.subheader("Performance par Catégorie (MRR)")
        df_group = df_retrieval.groupby("Catégorie", as_index=False)["MRR"].mean()
        fig = px.bar(df_group, x="Catégorie", y="MRR", color="Catégorie", text_auto='.2f')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Voir les données brutes"):
            st.dataframe(df_retrieval, use_container_width=True)

# ─── TAB 2 : ANSWER (LLM-as-a-judge) ───
with tab2:
    st.header("Évaluation de la Génération (LLM-as-a-judge)")
    st.write("Utilise un modèle plus puissant pour noter factuellement les réponses du RAG.")
    
    if st.button("▶ Lancer l'évaluation des Réponses", type="primary"):
        progress_text = "Jugement LLM en cours (peut être long)..."
        my_bar = st.progress(0, text=progress_text)
        
        ans_results = []
        for i, test in enumerate(tests):
            res = evaluate_answer(test)
            ans_results.append({
                "Catégorie": test.category,
                "Accuracy": res.accuracy,
                "Completeness": res.completeness,
                "Relevance": res.relevance,
                "Overall": res.overall
            })
            # Respecter les rate limits de l'API gratuite Groq
            time.sleep(1) 
            my_bar.progress((i + 1) / len(tests), text=f"Jugement de la réponse {i+1}/{len(tests)}")
            
        df_ans = pd.DataFrame(ans_results)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{df_ans['Accuracy'].mean():.1f}/5")
        col2.metric("Completeness", f"{df_ans['Completeness'].mean():.1f}/5")
        col3.metric("Relevance", f"{df_ans['Relevance'].mean():.1f}/5")
        col4.metric("Score Global", f"{df_ans['Overall'].mean():.1f}/5")
        
        st.subheader("Distribution des Scores Globaux")
        fig2 = px.box(df_ans, x="Catégorie", y="Overall", points="all", color="Catégorie")
        st.plotly_chart(fig2, use_container_width=True)

# ─── TAB 3 : CHAT INTERACTIF ───
with tab3:
    st.header("Chat avec le Système RAG")
    
    # Initialiser l'historique du chat dans la session Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input utilisateur
    if prompt := st.chat_input("Pose une question sur tes documents (ex: politique de congés)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche et réflexion..."):
                try:
                    reponse_rag = answer_question(prompt)
                    st.markdown(reponse_rag)
                    st.session_state.messages.append({"role": "assistant", "content": reponse_rag})
                except Exception as e:
                    st.error(f"Erreur de génération : {e}")