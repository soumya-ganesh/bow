!pip install nltk
import streamlit as st
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Bag of Words Visualizer", layout="wide")

st.title("🧠 Bag of Words Visualizer")

# --- Input Section ---
st.subheader("Enter Three Documents")
doc1 = st.text_area("Document 1")
doc2 = st.text_area("Document 2")
doc3 = st.text_area("Document 3")

if st.button("Generate Bag of Words"):
    if not doc1 or not doc2 or not doc3:
        st.warning("Please fill in all three documents.")
    else:
        docs = [word_tokenize(doc.lower()) for doc in [doc1, doc2, doc3]]
        vocab = sorted(set(word for doc in docs for word in doc if word.isalpha()))

        bow = []
        for doc in docs:
            freq = [doc.count(word) for word in vocab]
            bow.append(freq)

        # Display Tokens
        st.subheader("Tokens in Each Document")
        for i, tokens in enumerate(docs, start=1):
            st.write(f"**Document {i}:** {tokens}")

        # Vocabulary
        st.subheader("Vocabulary of Unique Words")
        st.write(", ".join(vocab))

        # Bag of Words Table
        st.subheader("Bag of Words Table")
        import pandas as pd
        df = pd.DataFrame({
            "Word": vocab,
            "Doc1": bow[0],
            "Doc2": bow[1],
            "Doc3": bow[2]
        })
        st.dataframe(df)

