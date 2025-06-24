import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# タブ名・アプリタイトルを好きなものに変更可能
st.set_page_config(page_title="教育QAチャット", layout="centered")
st.title("学校Q&A自動検索チャット")

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

# チャット入力欄
user_input = st.text_input("知りたいこと・悩みを入力してください", key="user_input")

# 入力例を表示
st.markdown("""
##### 入力例
- 例1：「不登校」
- 例2：「友人とのトラブルがあったときの対応は？」
""")

if user_input:
    with st.spinner("検索中..."):
        tfidf = TfidfVectorizer(tokenizer=tokenize)
        tfidf_matrix = tfidf.fit_transform(corpus + [user_input])
        sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        top_idx = sims.argsort()[-5:][::-1]
        best_idx = top_idx[0]
        best_Q = df.iloc[best_idx]["質問"]
        best_A = df.iloc[best_idx]["回答"]
        st.session_state.history.append(("ユーザー", user_input))
        st.session_state.history.append(("AI", f"おすすめQ&A：\n\n**Q:** {best_Q}\n\n**A:** {best_A}"))
    # チャット履歴表示
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
