# ⌨️ Autocorrect Keyboard — Next Word Prediction

A predictive keyboard system that anticipates the next word in a sentence using contextual information from preceding words, implemented with N-gram language modeling.

👉 **Live Demo:** [https://autocorrectkeyboard.app](https://autocorrectkeyboard-e5k.streamlit.app/)

## Problem Statement
Build an intuitive keyboard system that suggests the most likely next word as a user types, improving typing speed and accuracy — similar to smartphone keyboard autocomplete.

## Approach
Implemented a statistical **N-gram language model**:
1. **Corpus** — NLTK's Reuters corpus (54,000+ financial news sentences).
2. **Preprocessing** — lowercased text, removed punctuation/non-alphabetic tokens.
3. **Trigram model** — for every pair of consecutive words `(w1, w2)`, counted which word `w3` most often followed, using a `Counter`.
4. **Bigram fallback** — used when a trigram pair hasn't been seen before, falling back to single-word context.
5. **Prediction** — given input text, returns the top-N most likely next words, trigram first, bigram fallback second.
6. **Deployment** — Streamlit app with live suggestions and clickable word-completion buttons.

## Why N-grams (vs. RNN/LSTM)
N-grams were chosen as the primary implementation for speed, simplicity, and interpretability — they train in seconds with no GPU and are easy to explain and debug. An LSTM/RNN version can be added later as an extension for stronger long-range context modeling.

## Model Stats
- Vocabulary: ~28,700 unique words
- Trigram patterns learned: ~318,000 unique word-pairs

## Tech Stack
Python, NLTK, Streamlit

## How to Run

```bash
pip install streamlit nltk
python -m streamlit run app.py
```

Ensure `trigram_model.pkl` and `bigram_model.pkl` are in the same folder as `app.py`.

## Project Structure
```
autocorrect-app/
├── app.py                 # Streamlit deployment app
├── trigram_model.pkl      # trained trigram frequency model
├── bigram_model.pkl       # trained bigram frequency model (fallback)
```

## Example
Input: `"the company said"` → Suggestions: `it`, `the`, `its`

## Future Improvements
- Train an LSTM/RNN model for deeper context understanding
- Support for autocorrect (typo correction), not just next-word prediction
- Train on a more conversational corpus for everyday phrasing

## Author
Meghana Yara — B.Tech CSE (AI/ML), PVPSIT
