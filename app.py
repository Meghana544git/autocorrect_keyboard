import streamlit as st
import pickle

st.set_page_config(page_title="Autocorrect Keyboard", page_icon="⌨️", layout="centered")

# ---------- Load trained models ----------
with open("trigram_model.pkl", "rb") as f:
    trigram_model = pickle.load(f)

with open("bigram_model.pkl", "rb") as f:
    bigram_model = pickle.load(f)

# ---------- Prediction logic ----------
def predict_next_word(text, top_n=3):
    words = [w.lower() for w in text.split() if w.isalpha()]

    if len(words) >= 2:
        w1, w2 = words[-2], words[-1]
        candidates = trigram_model.get((w1, w2))
        if candidates:
            return [w for w, _ in candidates.most_common(top_n)]

    if len(words) >= 1:
        w1 = words[-1]
        candidates = bigram_model.get(w1)
        if candidates:
            return [w for w, _ in candidates.most_common(top_n)]

    return []

# ---------- Session state (keeps typed text across reruns) ----------
if "text" not in st.session_state:
    st.session_state.text = "the company said"

# ---------- Header ----------
st.title("⌨️ Autocorrect Keyboard")
st.caption("A smart keyboard that predicts your next word using N-gram language modeling, trained on the Reuters news corpus.")

st.divider()

# ---------- Text input ----------
st.session_state.text = st.text_area(
    "Type your sentence:",
    value=st.session_state.text,
    height=100,
)

suggestions = predict_next_word(st.session_state.text, top_n=3)

st.subheader("💡 Suggested next words")

if suggestions:
    cols = st.columns(len(suggestions))
    for col, word in zip(cols, suggestions):
        with col:
            if st.button(f"➕ {word}", use_container_width=True):
                st.session_state.text = st.session_state.text.rstrip() + " " + word
                st.rerun()
else:
    st.warning("No suggestion available — try typing a more common word or phrase.")

st.divider()

# ---------- Sidebar info ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses a **trigram + bigram fallback model** built from the "
        "NLTK Reuters corpus (54,000+ sentences of financial news text)."
    )
    st.write("**How it works:**")
    st.markdown(
        "- Looks at your last 2 words\n"
        "- Finds the most common word that followed that pair in the training data\n"
        "- Falls back to the last 1 word if no match is found"
    )
    st.write(f"**Vocabulary learned:** {len(bigram_model):,} unique words")
    st.write(f"**Word-pairs learned:** {len(trigram_model):,} trigram patterns")