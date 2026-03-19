"""
📚 Léo — Bibliothécaire IA (Streamlit + HuggingFace Spaces)
============================================================
Stack :
- LLM        : meta-llama/Llama-3.1-8B-Instruct via HF Inference API
- Streaming  : InferenceClient stream → st.write_stream
- Tool       : search_book() → Open Library API (gratuit, sans clé)
- Couvertures: Open Library Covers API
- Audio OUT  : gTTS → st.audio autoplay
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import json
import re
import base64
import tempfile
import requests
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv(override=True)  # charge .env en local (ignoré sur HF Spaces)
from gtts import gTTS
from huggingface_hub import InferenceClient

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Léo — Bibliothécaire IA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Initialisation client ─────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL    = "meta-llama/Llama-3.1-8B-Instruct"

@st.cache_resource
def get_client():
    return InferenceClient(model=MODEL, token=HF_TOKEN)

client = get_client()

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es Léo, un bibliothécaire passionné et érudit avec 20 ans d'expérience.
Tu aides les utilisateurs à :
- Trouver des livres selon leurs goûts, humeurs ou besoins
- Découvrir des auteurs et des genres littéraires
- Obtenir des résumés, analyses et recommandations de livres
- Chercher des informations précises sur un titre ou un auteur

Ton style : chaleureux, enthousiaste pour la littérature, pédagogue.
Tu réponds toujours en français sauf si l'utilisateur parle une autre langue.

Quand l'utilisateur mentionne un titre ou demande un livre précis, tu DOIS appeler
l'outil search_book en répondant UNIQUEMENT avec ce JSON (rien d'autre avant) :
{"tool": "search_book", "args": {"title": "<titre>", "author": "<auteur ou vide>"}}

Sinon, réponds normalement en texte.
"""

WELCOME = (
    "Bonjour et bienvenue ! 📚 Je suis **Léo**, votre bibliothécaire IA.\n\n"
    "Je peux vous aider à :\n"
    "- **Trouver un livre** selon vos goûts ou votre humeur\n"
    "- **Découvrir des auteurs** et des genres littéraires\n"
    "- **Obtenir un résumé** ou une analyse d'un ouvrage\n"
    "- **Recommander des lectures** personnalisées\n\n"
    "Que puis-je faire pour vous aujourd'hui ? 🌟"
)

# ── Open Library Tool ─────────────────────────────────────────────────────────
def search_book(title: str, author: str = "") -> dict:
    query = f"{title} {author}".strip().replace(" ", "+")
    url   = (
        f"https://openlibrary.org/search.json"
        f"?q={query}&limit=1&fields=title,author_name,first_publish_year,cover_i,subject"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if not data.get("docs"):
            return {"found": False, "message": f"Aucun résultat pour '{title}'"}

        doc      = data["docs"][0]
        cover_id = doc.get("cover_i")
        return {
            "found":     True,
            "title":     doc.get("title", title),
            "author":    ", ".join(doc.get("author_name", ["Inconnu"])),
            "year":      doc.get("first_publish_year", "?"),
            "cover_url": f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None,
            "subjects":  doc.get("subject", [])[:5],
        }
    except Exception as e:
        return {"found": False, "message": str(e)}


def fetch_cover(cover_url: str):
    if not cover_url:
        return None
    try:
        resp = requests.get(cover_url, timeout=10)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content))
    except Exception:
        pass
    return None


# ── TTS ───────────────────────────────────────────────────────────────────────
def make_audio_b64(message: str) -> str:
    """Génère un MP3 gTTS et retourne le base64 pour autoplay HTML."""
    clean = re.sub(r"[*_`#>\[\]()\{\}]", "", message)
    clean = re.sub(r"https?://\S+", "", clean).strip()[:500]
    tts   = gTTS(text=clean, lang="fr")
    buf   = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def autoplay_audio(b64: str):
    """Injecte un lecteur audio HTML autoplay invisible dans la page."""
    html = f"""
    <audio autoplay style="display:none">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Détection tool call ───────────────────────────────────────────────────────
TOOL_RE = re.compile(r'\{"tool"\s*:\s*"search_book".*?\}', re.DOTALL)

def parse_tool_call(text: str) -> dict | None:
    m = TOOL_RE.search(text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ── LLM streaming ─────────────────────────────────────────────────────────────
def stream_response(messages: list):
    """
    Générateur de tokens.
    Détecte un tool call JSON → exécute l'outil → relance le stream.
    Retourne aussi la couverture trouvée.
    """
    reply      = ""
    tool_fired = False
    cover_img  = None

    stream = client.chat_completion(messages=messages, max_tokens=1024, stream=True)

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        reply += delta
        if not tool_fired and parse_tool_call(reply):
            tool_fired = True
            break
        if not tool_fired:
            yield delta, None

    if tool_fired:
        tc        = parse_tool_call(reply)
        args      = tc.get("args", {})
        book_info = search_book(**args)

        if book_info.get("found"):
            cover_img = fetch_cover(book_info.get("cover_url"))

        messages.append({"role": "assistant", "content": reply})
        messages.append({
            "role": "user",
            "content": (
                f"[Résultat de la recherche]\n{json.dumps(book_info, ensure_ascii=False)}\n\n"
                "Présente ce livre à l'utilisateur de façon enthousiaste."
            ),
        })

        reply  = ""
        stream = client.chat_completion(messages=messages, max_tokens=1024, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            yield delta, cover_img


# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Crimson+Text:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'Crimson Text', Georgia, serif;
    background-color: #f5efe0;
    color: #2c1810;
}

/* En-tête */
.leo-header {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    border-bottom: 3px double #c9a84c;
    margin-bottom: 1.5rem;
}
.leo-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    color: #8b4513;
    letter-spacing: 0.05em;
    margin: 0;
}
.leo-header p {
    font-style: italic;
    color: #6b7c5e;
    font-size: 1.1rem;
    margin-top: 0.3rem;
}

/* Bulles de chat */
.chat-user {
    background: #e8d5b0;
    border-left: 4px solid #c9a84c;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 2rem 0.5rem 0;
    font-size: 1.05rem;
}
.chat-assistant {
    background: #fdf6e3;
    border-left: 4px solid #8b4513;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0 0.5rem 2rem;
    font-size: 1.05rem;
}
.chat-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.user-label   { color: #c9a84c; }
.leo-label    { color: #8b4513; }

/* Couverture */
.cover-box {
    border: 3px solid #c9a84c;
    border-radius: 4px;
    padding: 0.5rem;
    background: #fdf6e3;
    text-align: center;
}
.cover-title {
    font-family: 'Playfair Display', serif;
    color: #8b4513;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-style: italic;
}
.quote-box {
    text-align: center;
    font-style: italic;
    color: #6b7c5e;
    font-size: 0.95rem;
    margin-top: 1.5rem;
    padding: 1rem;
    border-top: 1px solid #c9a84c;
}

/* Input */
.stTextInput > div > div > input {
    background: #fdf6e3 !important;
    border: 1px solid #c9a84c !important;
    border-radius: 4px !important;
    font-family: 'Crimson Text', serif !important;
    font-size: 1.05rem !important;
    color: #2c1810 !important;
}

/* Boutons */
.stButton > button {
    background: #8b4513 !important;
    color: #f5efe0 !important;
    border: none !important;
    font-family: 'Playfair Display', serif !important;
    letter-spacing: 0.05em !important;
    border-radius: 4px !important;
}
.stButton > button:hover {
    background: #c9a84c !important;
    color: #2c1810 !important;
}

footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="leo-header">
    <h1>📚 Léo — Bibliothécaire IA</h1>
    <p>Votre guide littéraire personnel · Llama 3.1 8B · Open Library</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
if "cover" not in st.session_state:
    st.session_state.cover = None
if "cover_info" not in st.session_state:
    st.session_state.cover_info = None

# ── Layout principal ──────────────────────────────────────────────────────────
col_chat, col_side = st.columns([3, 2])

with col_chat:
    # ── Affichage de l'historique ─────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    <div class="chat-label user-label">Vous</div>
                    {msg['content']}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-assistant">
                    <div class="chat-label leo-label">📚 Léo</div>
                    {msg['content']}
                </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Zone de saisie ────────────────────────────────────────────────────────
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                label="Votre message",
                placeholder="Demandez un livre, un auteur, une recommandation...",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Envoyer →")

    if st.button("🗑️ Nouvelle conversation"):
        st.session_state.messages  = [{"role": "assistant", "content": WELCOME}]
        st.session_state.cover      = None
        st.session_state.cover_info = None
        st.rerun()

with col_side:
    # ── Couverture ────────────────────────────────────────────────────────────
    if st.session_state.cover:
        st.markdown('<div class="cover-box">', unsafe_allow_html=True)
        st.image(st.session_state.cover, use_container_width=True)
        if st.session_state.cover_info:
            info = st.session_state.cover_info
            st.markdown(
                f'<div class="cover-title">{info.get("title","")}'
                f'<br><small>{info.get("author","")} · {info.get("year","")}</small></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cover-box" style="height:300px;display:flex;align-items:center;
             justify-content:center;color:#c9a84c;font-size:4rem;">
            📖
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="quote-box">
        « Un livre est un jardin qu'on porte dans sa poche. »<br>
        <small>— Proverbe arabe</small>
    </div>
    """, unsafe_allow_html=True)

# ── Traitement du message envoyé ──────────────────────────────────────────────
if submitted and user_input.strip():
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Construire les messages pour le LLM
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages:
        llm_messages.append({"role": m["role"], "content": m["content"]})

    # Zone de streaming dans la colonne chat
    with col_chat:
        with st.container():
            st.markdown("""
            <div class="chat-assistant">
                <div class="chat-label leo-label">📚 Léo</div>
            """, unsafe_allow_html=True)

            # Streaming token par token
            reply     = ""
            cover_img = None
            placeholder = st.empty()

            for delta, maybe_cover in stream_response(llm_messages):
                reply += delta
                placeholder.markdown(reply + "▌")
                if maybe_cover is not None:
                    cover_img = maybe_cover

            placeholder.markdown(reply)
            st.markdown("</div>", unsafe_allow_html=True)

    # Sauvegarder la réponse
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Mettre à jour la couverture
    if cover_img:
        st.session_state.cover = cover_img
        # Récupérer les infos du dernier tool call
        for m in reversed(llm_messages):
            if m["role"] == "user" and "[Résultat de la recherche]" in m["content"]:
                try:
                    info_str = m["content"].split("\n")[1]
                    st.session_state.cover_info = json.loads(info_str)
                except Exception:
                    pass
                break

    # Audio autoplay
    try:
        b64 = make_audio_b64(reply)
        autoplay_audio(b64)
    except Exception as e:
        print(f"TTS error: {e}")

    st.rerun()