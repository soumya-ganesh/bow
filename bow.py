import streamlit as st
import pandas as pd
import math

st.set_page_config(page_title="Bag of Words Visualizer", layout="wide")

st.title("Bag of Words and TFIDF")

# --- Helper to left-align dataframe ---
def show_left_aligned(df):
    """Display a dataframe with all columns left aligned"""
    st.markdown(
        df.style.set_table_styles(
            [{"selector": "th, td", "props": [("text-align", "center")]}]
        ).to_html(),
        unsafe_allow_html=True
    )

# --- Input Section ---
st.subheader("Enter Three Documents")
doc1 = st.text_area("Document 1")
doc2 = st.text_area("Document 2")
doc3 = st.text_area("Document 3")

if st.button("Generate Tables"):
    if not doc1 or not doc2 or not doc3:
        st.warning("Please fill in all three documents.")
    else:
        # Tokenize by splitting on spaces
        docs = [doc.lower().split() for doc in [doc1, doc2, doc3]]

        # Step 1: Vocabulary
        vocab = sorted(set(word for doc in docs for word in doc if word.isalpha()))

        # Step 2: Bag of Words (raw frequency)
        bow = []
        for doc in docs:
            freq = [doc.count(word) for word in vocab]
            bow.append(freq)

        # Step 3: Term Frequency (TF) — using same BoW values
        tf = []
        for doc in bow:
            total_words = sum(doc)
            tf_doc = [round(f / total_words, 3) if total_words else 0 for f in doc]
            tf.append(tf_doc)

        # Display Tokens
        st.subheader("🧩 Tokens in Each Document")
        for i, tokens in enumerate(docs, start=1):
            st.write(f"**Document {i}:** {tokens}")

        # Vocabulary
        st.subheader("📘 Vocabulary of Unique Words")
        st.write(", ".join(vocab))

        # Combined Bag of Words + Term Frequency Table
        st.subheader("📊 Bag of Words / Term Frequency Table")
        combined_data = []
        for i in range(len(vocab)):
            combined_data.append({
                "Word": vocab[i],
                "Doc1 (Count)": bow[0][i],
                "Doc1 (TF)": tf[0][i],
                "Doc2 (Count)": bow[1][i],
                "Doc2 (TF)": tf[1][i],
                "Doc3 (Count)": bow[2][i],
                "Doc3 (TF)": tf[2][i],
            })
        combined_df = pd.DataFrame(combined_data).set_index("Word")
        show_left_aligned(combined_df)

        # Step 4: Document Frequency (DF)
        st.subheader("📄 Document Frequency (DF) Table")
        df_counts = []
        for word in vocab:
            df_counts.append(sum(1 for doc in docs if word in doc))
        df_table = pd.DataFrame({"Word": vocab, "Document Frequency": df_counts}).set_index("Word")
        show_left_aligned(df_table)

        # Step 5: Inverse Document Frequency (IDF)
        st.subheader("🔢 Inverse Document Frequency (IDF) Table")
        num_docs = len(docs)
        idf = {word: f"{num_docs}/{df_counts[i]}" for i, word in enumerate(vocab)}
        idf_table = pd.DataFrame({
            "Word": vocab,
            "IDF Formula (No Calculation)": [idf[word] for word in vocab]
        }).set_index("Word")
        show_left_aligned(idf_table)

        # Step 6: TF-IDF (as expression, not calculated)
        st.subheader("🧮 TF × log(IDF) Formula Table (Using TF from BoW Table)")
        tfidf_display = []
        for i in range(len(docs)):
            tfidf_display.append([
                f"{tf[i][j]} × log({idf[vocab[j]]})"
                for j in range(len(vocab))
            ])

        tfidf_df = pd.DataFrame(tfidf_display, columns=vocab, index=["Doc1", "Doc2", "Doc3"])
        show_left_aligned(tfidf_df)
