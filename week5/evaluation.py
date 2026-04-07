"""
evaluation.py — Dashboard Gradio pour l'évaluation RAG
Interface identique au dashboard du cours (RAG Evaluation Dashboard)
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from evaluation.test import load_tests
from evaluation.eval import evaluate_retrieval, evaluate_answer

# ─── CHARGEMENT DES TESTS (une fois) ──────────────────────────────────────────
tests = load_tests()
print(f"✅ {len(tests)} questions de test chargées")


# ─── ÉVALUATION RETRIEVAL ─────────────────────────────────────────────────────

def run_retrieval_eval():
    """
    Lance l'évaluation de retrieval sur tous les tests.
    Retourne: markdown stats + figure plotly
    """
    results = []
    
    for test in tests:
        res = evaluate_retrieval(test)
        results.append({
            "category": test.category,
            "question": test.question[:60] + "...",
            "mrr": res.mrr,
            "ndcg": res.ndcg,
            "keywords_coverage": res.keywords_coverage,
        })
    
    df = pd.DataFrame(results)
    
    # Moyennes globales
    mean_mrr = df["mrr"].mean()
    mean_ndcg = df["ndcg"].mean()
    mean_coverage = df["keywords_coverage"].mean()
    total_tests = len(tests)
    
    # Breakdown par catégorie
    category_df = df.groupby("category", as_index=False)["mrr"].mean()
    
    # Figure plotly (comme dans le cours)
    fig = px.bar(
        category_df,
        x="category",
        y="mrr",
        title="Average MRR by Category",
        color="mrr",
        color_continuous_scale="RdYlGn",
        labels={"mrr": "Mean MRR", "category": "Category"},
    )
    fig.update_layout(yaxis_range=[0, 1], showlegend=False)
    
    # Couleurs pour les métriques (vert si bon, orange si moyen, rouge si mauvais)
    def color_score(val):
        if val >= 0.75: return "🟢"
        elif val >= 0.55: return "🟡"
        else: return "🔴"
    
    stats_md = f"""
## 📊 Overall Results — {total_tests} tests

| Métrique | Score | Status |
|----------|-------|--------|
| **Mean Reciprocal Rank (MRR)** | **{mean_mrr:.4f}** | {color_score(mean_mrr)} |
| **Normalized DCG (nDCG)** | **{mean_ndcg:.4f}** | {color_score(mean_ndcg)} |
| **Keyword Coverage** | **{mean_coverage:.1%}** | {color_score(mean_coverage)} |

### Interprétation
- **MRR > 0.75** → Excellent retrieval
- **MRR 0.55-0.75** → Correct, peut être amélioré  
- **MRR < 0.55** → Problème — change chunk_size ou le modèle d'embedding

### Résultats par catégorie
{category_df.to_markdown(index=False, floatfmt=".4f")}
"""
    
    return stats_md, fig


# ─── ÉVALUATION ANSWER (LLM Judge) ───────────────────────────────────────────

def run_answer_eval(progress=gr.Progress()):
    """
    Lance l'évaluation des réponses via LLM-as-a-judge.
    Plus lent (appel LLM par question).
    """
    results = []
    
    for i, test in enumerate(tests):
        progress((i + 1) / len(tests), desc=f"Évaluation {i+1}/{len(tests)}...")
        
        score = evaluate_answer(test)
        results.append({
            "category": test.category,
            "question": test.question[:50] + "...",
            "accuracy": score.accuracy,
            "completeness": score.completeness,
            "relevance": score.relevance,
            "overall": score.overall,
        })
    
    df = pd.DataFrame(results)
    
    # Moyennes
    mean_accuracy = df["accuracy"].mean()
    mean_completeness = df["completeness"].mean()
    mean_relevance = df["relevance"].mean()
    mean_overall = df["overall"].mean()
    
    # Figure: scores par catégorie
    cat_df = df.groupby("category")[["accuracy", "completeness", "relevance"]].mean().reset_index()
    cat_melted = cat_df.melt(id_vars="category", var_name="dimension", value_name="score")
    
    fig = px.bar(
        cat_melted,
        x="category",
        y="score",
        color="dimension",
        barmode="group",
        title="Answer Quality by Category",
        labels={"score": "Score /5", "category": "Category"},
        color_discrete_map={
            "accuracy": "#3B82F6",
            "completeness": "#10B981",
            "relevance": "#F59E0B",
        },
    )
    fig.update_layout(yaxis_range=[0, 5])
    
    def color_score(val, max_val=5):
        ratio = val / max_val
        if ratio >= 0.8: return "🟢"
        elif ratio >= 0.6: return "🟡"
        else: return "🔴"
    
    stats_md = f"""
## 🤖 Answer Quality — LLM Judge ({len(tests)} questions)

| Dimension | Score | Status |
|-----------|-------|--------|
| **Accuracy** (exactitude) | **{mean_accuracy:.2f}/5** | {color_score(mean_accuracy)} |
| **Completeness** (complétude) | **{mean_completeness:.2f}/5** | {color_score(mean_completeness)} |
| **Relevance** (pertinence) | **{mean_relevance:.2f}/5** | {color_score(mean_relevance)} |
| **Overall** | **{mean_overall:.2f}/5** | {color_score(mean_overall)} |

### Résultats complets par question
{df[["category", "question", "accuracy", "completeness", "relevance", "overall"]].to_markdown(index=False, floatfmt=".2f")}
"""
    
    return stats_md, fig


# ─── INTERFACE GRADIO ─────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="RAG Evaluation Dashboard",
        theme=gr.themes.Soft(),
        css="""
        .metric-box { border-radius: 8px; padding: 16px; margin: 8px; }
        h1 { color: #1e293b; }
        """,
    ) as demo:
        
        gr.Markdown("""
        # 📊 RAG Evaluation Dashboard
        Évalue la qualité de retrieval et de réponse de ton système RAG
        """)
        
        gr.Markdown(f"**{len(tests)} questions de test** | Utilise `tests.jsonl` pour ajouter/modifier les tests")
        
        # ── TAB 1: RETRIEVAL EVALUATION ──
        with gr.Tab("🔍 Retrieval Evaluation"):
            gr.Markdown("""
            Mesure la qualité de la **récupération de documents** :
            - **MRR** : Le bon document est-il récupéré en premier ?
            - **nDCG** : Les documents les plus pertinents sont-ils bien classés ?
            - **Keyword Coverage** : Les mots-clés importants sont-ils dans les chunks récupérés ?
            """)
            
            retrieval_btn = gr.Button("▶ Run Retrieval Evaluation", variant="primary", size="lg")
            
            with gr.Row():
                retrieval_stats = gr.Markdown("*Clique sur le bouton pour lancer l'évaluation...*")
            
            retrieval_plot = gr.Plot()
            
            retrieval_btn.click(
                fn=run_retrieval_eval,
                outputs=[retrieval_stats, retrieval_plot],
            )
        
        # ── TAB 2: ANSWER EVALUATION (LLM Judge) ──
        with gr.Tab("🤖 Answer Evaluation (LLM Judge)"):
            gr.Markdown("""
            Évalue la **qualité des réponses** via un LLM juge :
            - **Accuracy** : Les faits sont-ils corrects par rapport à la référence ?
            - **Completeness** : La réponse est-elle complète ?
            - **Relevance** : La réponse est-elle bien ciblée sur la question ?
            
            ⚠️ *Plus lent — fait un appel LLM par question*
            """)
            
            answer_btn = gr.Button("▶ Run Answer Evaluation", variant="primary", size="lg")
            
            with gr.Row():
                answer_stats = gr.Markdown("*Clique sur le bouton pour lancer l'évaluation...*")
            
            answer_plot = gr.Plot()
            
            answer_btn.click(
                fn=run_answer_eval,
                outputs=[answer_stats, answer_plot],
            )
        
        # ── TAB 3: TEST INTERACTIF ──
        with gr.Tab("💬 Test interactif"):
            gr.Markdown("Teste le système RAG avec une question personnalisée")
            
            question_input = gr.Textbox(
                label="Ta question",
                placeholder="Ex: Quelle est la politique de congés de l'entreprise ?",
                lines=2,
            )
            ask_btn = gr.Button("Poser la question", variant="primary")
            answer_output = gr.Textbox(label="Réponse RAG", lines=6)
            
            ask_btn.click(
                fn=lambda q: __import__("answer").answer_question(q),
                inputs=[question_input],
                outputs=[answer_output],
            )
    
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )
