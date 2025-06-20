import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI QA検索チャット", layout="centered")
st.title("AI QAレコメンド・チャット（日本語強化版）")

QA_FILE = "QA_索引付きQA集.xlsx"
@st.cache_data
def load_qa():
    return pd.read_excel(QA_FILE, sheet_name="全件データ").dropna(subset=["質問", "回答"])

df = load_qa()
corpus = (df["質問"].fillna("") + " " + df["回答"].fillna("")).tolist()

# 日本語の分かち書き
tokenizer = Tokenizer(wakati=True)
def tokenize(text):
    return list(tokenizer.tokenize(str(text)))

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("知りたいこと・悩みを入力してください", key="user_input")

if user_input:
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    tfidf_matrix = tfidf.fit_transform(corpus + [user_input])
    sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_idx = sims.argsort()[-5:][::-1]
    best_idx = top_idx[0]
    best_Q = df.iloc[best_idx]["質問"]
    best_A = df.iloc[best_idx]["回答"]
    st.session_state.history.append(("ユーザー", user_input))
    st.session_state.history.append(("AI", f"おすすめQ&A：\n\n**Q:** {best_Q}\n\n**A:** {best_A}"))
    for role, msg in st.session_state.history:
        if role == "ユーザー":
            st.markdown(f"🧑‍💻 **あなた:** {msg}")
        else:
            st.markdown(f"🤖 **AI:** {msg}")
    st.markdown("---")
    st.markdown("### 他にもこんなQ&Aがあります")
    for idx in top_idx[1:]:
        st.markdown(f"- **Q:** {df.iloc[idx]['質問']}\n    \n**A:** {df.iloc[idx]['回答']}")
else:
    for role, msg in st.session_state.history:
        if role == "ユーザー":
            st.markdown(f"🧑‍💻 **あなた:** {msg}")
        else:
            st.markdown(f"🤖 **AI:** {msg}")

st.markdown("---\n\n*このアプリはQAエクセルから日本語形態素解析を使って検索・推薦しています*")
