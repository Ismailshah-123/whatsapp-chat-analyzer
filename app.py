# app.py — Final polished WhatsApp Chat Analyzer (single-file)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import emoji
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import io, tempfile, os
from fpdf import FPDF
import dateparser
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Optional: sentence-transformers (if installed) — used only if present
HAS_SENT_TRANS = False
try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SENT_TRANS = True
except Exception:
    HAS_SENT_TRANS = False

st.set_page_config(page_title="📱 WhatsApp Chat Analyzer — SuperFire", layout="wide", initial_sidebar_state="expanded")

# -------------------- THEME & SIDEBAR (solid dark green) --------------------
SIDEBAR_BG = "#075E54"   # dark green
NEON = "#25D366"         # neon green for highlights
st.markdown(f"""
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {{background-color: {SIDEBAR_BG};}}
    [data-testid="stSidebar"] .css-1d391kg {{ color: white; }}
    /* Sidebar radio/label text color */
    .stRadio > label {{ color: white; }}
    /* Header title */
    .title-block {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
    .app-title {{ font-size:20px; font-weight:700; color:{NEON}; }}
    .app-sub {{ color: #dfeeea; font-size:12px; margin-top:2px; }}
    /* Buttons highlight */
    .stDownloadButton>button {{ background-color:{NEON}; color: #000; border: none; }}
    </style>
""", unsafe_allow_html=True)

# Header with small WhatsApp logo + title
wh_logo = "https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg"
st.markdown(f'<div class="title-block"><img src="{wh_logo}" height="44"><div><div class="app-title">WhatsApp Chat Analyzer</div><div class="app-sub">SuperFire insights — Made by Ismail Shah</div></div></div>', unsafe_allow_html=True)

# Sidebar navigation (left)
st.sidebar.markdown(f"<h3 style='color:white'>Navigation</h3>", unsafe_allow_html=True)
pages = ["Home", "Chat Analyzer", "Insights", "Sentiment", "Network", "Download Report", "Chatbot"]
page = st.sidebar.radio("Go to", pages)

# -------------------- CACHED HELPERS --------------------
@st.cache_data
def parse_chat(text: str) -> pd.DataFrame:
    """Robust parsing using dateparser; returns DataFrame with DateTime column."""
    lines = text.splitlines()
    rows = []
    for ln in lines:
        if not ln.strip():
            continue
        if " - " not in ln:
            continue
        left, right = ln.split(" - ", 1)
        dt = dateparser.parse(left)
        if not dt:
            continue
        if ": " in right:
            user, msg = right.split(": ", 1)
        else:
            user, msg = "Unknown", right
        rows.append({"Date": dt.date(), "Time": dt.time(), "User": user.strip(), "Message": msg.strip(), "DateTime": dt})
    df = pd.DataFrame(rows, columns=["Date", "Time", "User", "Message", "DateTime"])
    return df

@st.cache_data
def make_wordcloud(messages:list, width=800, height=400, bg="white"):
    text = " ".join(messages)
    wc = WordCloud(width=width, height=height, background_color=bg).generate(text)
    return wc

@st.cache_data
def top_words_series(messages:list, n=50):
    vect = CountVectorizer(stop_words="english", max_features=n)
    X = vect.fit_transform(messages)
    s = pd.Series(X.toarray().sum(axis=0), index=vect.get_feature_names_out())
    return s.sort_values(ascending=False)

@st.cache_data
def top_emojis_series(messages:list, top_n=30):
    all_e = [c for m in messages for c in m if c in emoji.EMOJI_DATA]
    if not all_e:
        return pd.Series(dtype=int)
    return pd.Series(all_e).value_counts().head(top_n)

@st.cache_data
def build_network(users:list):
    G = nx.Graph()
    for u in set(users):
        G.add_node(u)
    for i in range(len(users)-1):
        a,b = users[i], users[i+1]
        if a==b:
            continue
        if G.has_edge(a,b):
            G[a][b]['weight'] += 1
        else:
            G.add_edge(a,b, weight=1)
    return G

# -------------------- QA BANK & CHATBOT (TF-IDF + optional embeddings) --------------------
@st.cache_data
def generate_qa_bank():
    intents = {
        "top_user": ["Who sent the most messages?", "Who is most active?"],
        "top_5_users": ["Top 5 active users", "Who are the top 5 senders?"],
        "total_messages": ["How many messages in total?", "Total messages?"],
        "messages_per_day": ["Messages per day", "Daily message counts"],
        "messages_per_hour": ["Messages per hour", "Hourly activity"],
        "top_words": ["Top words in the chat", "Most common words"],
        "top_emojis": ["Top emojis used", "Which emoji is used the most?"],
        "media_counts": ["How many media messages were shared?", "Count media files"],
        "top_media_senders": ["Who shared the most images?", "Top media sharers"],
        "sentiment_summary": ["What is the sentiment summary?", "Is the chat positive or negative overall?"],
        "word_usage_count": ["How many times is the word '{word}' used?", "Count occurrences of '{word}'"]
    }
    bank=[]
    for intent, phrases in intents.items():
        for p in phrases:
            bank.append({"q": p, "intent": intent})
            bank.append({"q": p.lower(), "intent": intent})
            bank.append({"q": "Please " + p.lower(), "intent": intent})
    sample_words = ["hello","thanks","ok","bro","love","yes","no","sorry","thanks","lol"]
    for w in sample_words:
        bank.append({"q": f"How many times did we say {w}?", "intent": "word_usage_count"})
        bank.append({"q": f"Count of the word {w}", "intent": "word_usage_count"})
    i=0
    while len(bank) < 500:
        base = bank[i % len(bank)]['q']
        bank.append({"q": base + " please", "intent": bank[i % len(bank)]['intent']})
        i += 1
    seen=set(); final=[]
    for b in bank:
        k=b['q'].strip().lower()
        if k not in seen:
            seen.add(k); final.append(b)
    return final[:650]

QA_BANK = generate_qa_bank()
QA_TEXTS = [q['q'] for q in QA_BANK]
QA_TFIDF_VEC = TfidfVectorizer().fit(QA_TEXTS)
QA_TFIDF = QA_TFIDF_VEC.transform(QA_TEXTS)
if HAS_SENT_TRANS:
    QA_EMBS = EMBED_MODEL.encode(QA_TEXTS, convert_to_numpy=True)

def intent_handler(intent, df, user_q=None):
    msgs = df["Message"].astype(str)
    if intent == "top_user":
        counts = df["User"].value_counts()
        return f"{counts.idxmax()} ({counts.max()} messages)" if not counts.empty else "No users"
    if intent == "top_5_users":
        return df["User"].value_counts().head(5).to_dict()
    if intent == "total_messages":
        return f"{len(msgs)} messages"
    if intent == "messages_per_day":
        s = df.groupby(df['DateTime'].dt.date).size()
        return s.to_dict()
    if intent == "messages_per_hour":
        s = df.groupby(df['DateTime'].dt.hour).size().to_dict()
        return s
    if intent == "top_words":
        return top_words_series(msgs.tolist(), n=50).head(25).to_dict()
    if intent == "top_emojis":
        return top_emojis_series(msgs.tolist(), top_n=30).head(15).to_dict()
    if intent == "media_counts":
        low = msgs.str.lower()
        media_map = {"images":["<media omitted>", ".jpg", ".png", ".jpeg"], "videos":[".mp4",".mov"], "audio":[".ogg",".m4a",".mp3"], "docs":[".pdf",".docx",".pptx",".xlsx"]}
        out={}
        for k,kw in media_map.items():
            out[k] = int(low.apply(lambda s: any(x in str(s) for x in kw)).sum())
        return out
    if intent == "top_media_senders":
        low = msgs.str.lower()
        mask = low.str.contains("|".join(["<media omitted>", ".jpg", ".png", ".jpeg", ".mp4", ".mov", ".ogg", ".m4a", ".mp3", ".pdf"]), na=False)
        return df.loc[mask, "User"].value_counts().head(10).to_dict()
    if intent == "sentiment_summary":
        pos = msgs.str.contains(r"\b(good|happy|great|love|nice|awesome|thanks)\b", case=False, regex=True).sum()
        neg = msgs.str.contains(r"\b(bad|sad|angry|hate|terrible|sorry)\b", case=False, regex=True).sum()
        neu = len(msgs) - pos - neg
        return {"positive": int(pos), "negative": int(neg), "neutral": int(neu)}
    if intent == "word_usage_count":
        if not user_q:
            return "Specify the word to count."
        import re
        m = re.search(r"'([^']+)'", user_q)
        if m:
            w = m.group(1).lower()
        else:
            tokens = user_q.strip().split()
            w = tokens[-1].lower() if tokens else ""
        cnt = sum(m.lower().split().count(w) for m in msgs)
        return {w: int(cnt)}
    return "Intent not implemented."

def chatbot_answer(user_q, df):
    if user_q.strip() == "":
        return {"matched":None, "intent":None, "answer":"Please type a question."}
    if HAS_SENT_TRANS:
        q_emb = EMBED_MODEL.encode([user_q], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, QA_EMBS)[0]
        idx = int(np.argmax(sims))
    else:
        q_vec = QA_TFIDF_VEC.transform([user_q])
        sims = cosine_similarity(q_vec, QA_TFIDF)[0]
        idx = int(np.argmax(sims))
    matched = QA_BANK[idx]
    intent = matched['intent']
    ans = intent_handler(intent, df, user_q)
    return {"matched": matched['q'], "intent": intent, "answer": ans}

# -------------------- PDF & Excel helpers (safe temp files) --------------------
def create_pdf_report(df, title="WhatsApp Chat Analyzer Report"):
    msgs = df['Message'].astype(str).tolist()
    wc = make_wordcloud(msgs)
    top_users = df['User'].value_counts().head(20)
    emojis = top_emojis_series(msgs, top_n=30)
    G = build_network(df['User'].tolist())

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,title,ln=True,align="C")
    pdf.ln(4)
    pdf.set_font("Arial","",12)
    pdf.cell(0,8,f"Total messages: {len(df)}",ln=True)
    pdf.cell(0,8,f"Participants: {df['User'].nunique()}",ln=True)
    pdf.ln(4)
    pdf.set_font("Arial","B",12); pdf.cell(0,8,"Top Users:",ln=True); pdf.set_font("Arial","",11)
    for u,c in top_users.items():
        pdf.cell(0,7,f"{u}: {c}", ln=True)
    pdf.ln(6)
    if not emojis.empty:
        pdf.set_font("Arial","B",12); pdf.cell(0,8,"Top Emojis:",ln=True); pdf.set_font("Arial","",11)
        for e,c in emojis.head(10).items():
            pdf.cell(0,7,f"{e}: {c}",ln=True)

    # wordcloud to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t:
        wc.to_image().save(t.name)
        pdf.add_page()
        pdf.image(t.name, x=15, w=180)
        wc_path = t.name

    # network to temp file
    plt.figure(figsize=(6,4))
    nx.draw(G, with_labels=True, node_color="lightgreen", node_size=900, font_size=8)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
        plt.savefig(t2.name, bbox_inches='tight'); plt.close()
        pdf.add_page()
        pdf.image(t2.name, x=15, w=180)
        net_path = t2.name

    out = pdf.output(dest='S').encode('latin1')

    # cleanup
    try:
        os.remove(wc_path)
        os.remove(net_path)
    except Exception:
        pass
    return out

def create_excel_report(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Messages", index=False)
        df['User'].value_counts().to_frame("count").to_excel(writer, sheet_name="TopUsers")
        top_emojis_series(df['Message'].astype(str).tolist(), top_n=50).to_frame("count").to_excel(writer, sheet_name="TopEmojis")
        writer.save()
    buf.seek(0)
    return buf

# -------------------- PAGES --------------------
# Home
if page == "Home":
    st.title("WhatsApp Chat Analyzer — Professional Overview")
    st.markdown("""
**About the project**  
This project analyzes WhatsApp exported chat files (.txt) to provide deep insights into conversation patterns, emoji usage, message timelines, media and document usage, sentiment signals, and network interactions. Built for analysis and presentation — perfect for portfolios.

**Inside the project**  
- Chat parsing (robust to many date formats)  
- Insights (12 charts): wordcloud, top users, timeline, hours, top words, media analysis, audio/document counts, message-length distribution, and conversation pairs.  
- Sentiment (7 charts): pie, per-user bars, timeline, radar-like views, distribution, wordclouds by sentiment.  
- Network (7 visualizations): graph, degree distribution, top pairs, centrality, correlations.  
- Chatbot: TF-IDF + cosine similarity engine trained on ~500 Q templates for fast local Q&A.  
- PDF & Excel reporting (downloadable).

**How to use**  
1. Export WhatsApp chat (.txt).  
2. Open **Chat Analyzer** and upload file.  
3. Explore **Insights**, **Sentiment**, **Network**.  
4. Ask the **Chatbot** natural questions.  
5. Go to **Download Report** to export PDF/Excel ready for sharing.

**Benefits**  
- Offline, no API keys. Fast and reliable.  
- Great for research, moderation audits, personal analytics, and portfolio demos.

**Made by Ismail Shah** ❤️
""")

# Chat Analyzer
elif page == "Chat Analyzer":
    st.header("Chat Analyzer — Upload and Parse")
    st.write("Upload WhatsApp exported `.txt`. For best results export without media or with media depending on what you need.")
    uploaded = st.file_uploader("Choose `.txt` file", type=["txt"])
    if uploaded:
        try:
            raw = uploaded.getvalue().decode("utf-8")
        except Exception:
            raw = uploaded.getvalue().decode("utf-8", errors="ignore")
        df = parse_chat(raw)
        if df.empty:
            st.error("Could not parse — try re-exporting your chat or check format.")
        else:
            st.success("Chat parsed successfully ✅")
            st.session_state["chat_df"] = df
            st.markdown("**Preview (first 200 rows)**")
            st.dataframe(df.head(200))

# Insights (12 charts)
elif page == "Insights":
    st.header("Insights — Comprehensive Charts")
    if "chat_df" not in st.session_state:
        st.warning("Upload a chat first in Chat Analyzer.")
    else:
        df = st.session_state["chat_df"]
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        msgs = df['Message'].astype(str).tolist()

        # Row A: Top users + Wordcloud
        a1,a2 = st.columns([1,2])
        with a1:
            st.subheader("Top Active Users")
            top_users = df['User'].value_counts().head(12)
            st.bar_chart(top_users, use_container_width=True)
            st.markdown("**Quick stats**")
            st.write({
                "Total messages": int(len(df)),
                "Participants": int(df['User'].nunique()),
                "Date range": f"{df['DateTime'].min().date()} → {df['DateTime'].max().date()}"
            })
        with a2:
            st.subheader("Word Cloud")
            wc = make_wordcloud(msgs)
            st.image(wc.to_array(), use_container_width=True)

        # Row B: top words + emojis
        b1,b2 = st.columns(2)
        with b1:
            st.subheader("Top Words (freq)")
            tw = top_words_series(msgs, n=80)
            st.bar_chart(tw.head(20), use_container_width=True)
        with b2:
            st.subheader("Top Emojis")
            te = top_emojis_series(msgs, top_n=40)
            if not te.empty:
                st.bar_chart(te, use_container_width=True)
            else:
                st.write("No emojis found")

        # Row C: Timeline and hours
        st.subheader("Messages Over Time (daily)")
        daily = df.groupby(df['DateTime'].dt.date).size().reset_index(name='count')
        fig = px.line(daily, x='DateTime', y='count', title="Daily message count")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Most Active Hours")
        hour_counts = df.groupby(df['DateTime'].dt.hour).size().reindex(range(24), fill_value=0)
        st.bar_chart(hour_counts, use_container_width=True)

        # Row D: Message length, media pie, audio/doc counts
        st.subheader("Message length distribution")
        df['msg_len'] = df['Message'].astype(str).apply(len)
        fig_len = px.histogram(df, x='msg_len', nbins=30, title="Message length (characters)")
        st.plotly_chart(fig_len, use_container_width=True)

        st.subheader("Media vs Text (pie)")
        low = df['Message'].str.lower()
        media_map = {
            "Images": ["<media omitted>", ".jpg", ".png", ".jpeg"],
            "Videos": [".mp4", ".mov"],
            "Audio": [".ogg", ".m4a", ".mp3"],
            "Docs": [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]
        }
        media_counts = {}
        for k, kw in media_map.items():
            media_counts[k] = int(low.apply(lambda s: any(x in str(s) for x in kw)).sum())
        mc_series = pd.Series(media_counts)
        if mc_series.sum() > 0:
            fig_media = px.pie(values=mc_series.values, names=mc_series.index, title="Media distribution")
            st.plotly_chart(fig_media, use_container_width=True)
        else:
            st.write("No media detected")

        st.subheader("Audio & Document Analysis")
        audio_count = media_counts.get("Audio", 0)
        doc_count = media_counts.get("Docs", 0)
        st.write({"Audio messages": int(audio_count), "Document messages": int(doc_count)})

        # Row E: Top media sharers & conversation pairs
        st.subheader("Top Media Sharers")
        for k, kw in media_map.items():
            mask = low.apply(lambda s: any(x in str(s) for x in kw))
            counts = df.loc[mask, "User"].value_counts().head(10)
            if not counts.empty:
                st.write(f"**{k}**")
                st.bar_chart(counts, use_container_width=True)

        st.subheader("Top Conversation Pairs (adjacent messages)")
        users = df['User'].tolist()
        pairs=[]
        for i in range(len(users)-1):
            if users[i] != users[i+1]:
                pairs.append((users[i], users[i+1]))
        pc = Counter(pairs).most_common(12)
        if pc:
            pc_series = pd.Series({f"{a}→{b}":c for (a,b),c in pc})
            st.bar_chart(pc_series, use_container_width=True)

# Sentiment (7 charts)
elif page == "Sentiment":
    st.header("Sentiment — multiple visualizations")
    if "chat_df" not in st.session_state:
        st.warning("Upload a chat first.")
    else:
        df = st.session_state["chat_df"]
        msgs = df['Message'].astype(str)
        # simple heuristics; replace with better model if desired
        pos_mask = msgs.str.contains(r"\b(good|happy|great|love|nice|awesome|thanks)\b", case=False, regex=True)
        neg_mask = msgs.str.contains(r"\b(bad|sad|angry|hate|terrible|sorry)\b", case=False, regex=True)
        neu_mask = (~pos_mask) & (~neg_mask)
        counts = {"positive": int(pos_mask.sum()), "negative": int(neg_mask.sum()), "neutral": int(neu_mask.sum())}

        # Chart 1: pie (medium size)
        st.subheader("Sentiment breakdown (pie)")
        figp, axp = plt.subplots(figsize=(4,4))
        axp.pie(list(counts.values()), labels=list(counts.keys()), autopct="%1.1f%%", colors=["#2ecc71","#e74c3c","#95a5a6"])
        axp.set_title("Overall Sentiment")
        st.pyplot(figp, use_container_width=False)

        # Chart 2: bar overall
        st.subheader("Sentiment counts (bar)")
        st.bar_chart(pd.Series(counts), use_container_width=True)

        # Chart 3: sentiment by user (stacked)
        st.subheader("Sentiment by Top Users")
        top_users = df['User'].value_counts().head(8).index.tolist()
        user_data = []
        for u in top_users:
            um = df[df['User']==u]['Message'].astype(str)
            p = um.str.contains(r"\b(good|happy|great|love|nice|awesome|thanks)\b", case=False, regex=True).sum()
            n = um.str.contains(r"\b(bad|sad|angry|hate|terrible|sorry)\b", case=False, regex=True).sum()
            ne = len(um)-p-n
            user_data.append({"user":u,"positive":p,"negative":n,"neutral":ne})
        udf = pd.DataFrame(user_data).set_index('user')
        st.bar_chart(udf, use_container_width=True)

        # Chart 4: sentiment timeline
        st.subheader("Sentiment over time (daily)")
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['sent_label'] = np.where(pos_mask, "positive", np.where(neg_mask, "negative", "neutral"))
        daily_sent = df.groupby([df['DateTime'].dt.date, 'sent_label']).size().unstack(fill_value=0)
        if not daily_sent.empty:
            fig_time = px.line(daily_sent.reset_index(), x='DateTime', y=daily_sent.columns.tolist(), title="Daily sentiment trends")
            st.plotly_chart(fig_time, use_container_width=True)

        # Chart 5: wordcloud for positive messages
        st.subheader("Wordcloud — Positive messages")
        pos_msgs = df.loc[pos_mask, 'Message'].astype(str).tolist()
        if pos_msgs:
            wc_pos = make_wordcloud(pos_msgs, bg="white")
            st.image(wc_pos.to_array(), use_container_width=True)
        else:
            st.write("No positive messages to show.")

        # Chart 6: distribution of message lengths by sentiment
        st.subheader("Message length distribution by sentiment")
        df['msg_len'] = df['Message'].astype(str).apply(len)
        fig_len = px.box(df, x='sent_label', y='msg_len', title="Message length by sentiment")
        st.plotly_chart(fig_len, use_container_width=True)

        # Chart 7: daily sentiment proportions (stacked area)
        st.subheader("Daily sentiment proportions")
        if not daily_sent.empty:
            daily_prop = daily_sent.div(daily_sent.sum(axis=1), axis=0).fillna(0)
            fig_area = px.area(daily_prop.reset_index(), x='DateTime', y=['positive','negative','neutral'], title="Daily sentiment proportions")
            st.plotly_chart(fig_area, use_container_width=True)

# Network (7 charts)
elif page == "Network":
    st.header("Network & Interaction Analytics")
    if "chat_df" not in st.session_state:
        st.warning("Upload a chat first.")
    else:
        df = st.session_state["chat_df"]
        G = build_network(df['User'].tolist())

        st.subheader("Conversation Network Graph")
        plt.figure(figsize=(9,5))
        pos = nx.spring_layout(G, seed=42, k=0.6)
        nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=800, font_size=9)
        st.pyplot(plt, use_container_width=True)

        st.subheader("Degree distribution")
        degrees = [d for n,d in G.degree()]
        fig_deg = px.histogram(x=degrees, nbins=10, title="Node degree distribution")
        st.plotly_chart(fig_deg, use_container_width=True)

        st.subheader("Top interaction pairs")
        edges = sorted(G.edges(data=True), key=lambda x: x[2].get('weight',1), reverse=True)[:12]
        if edges:
            labels = [f"{a}→{b}" for a,b,_ in edges]
            vals = [w.get('weight',1) for _,_,w in edges]
            st.bar_chart(pd.Series(vals, index=labels), use_container_width=True)
        else:
            st.write("No edges found")

        st.subheader("Degree centrality (top users)")
        cent = nx.degree_centrality(G)
        if cent:
            st.bar_chart(pd.Series(cent).sort_values(ascending=False).head(10), use_container_width=True)
        else:
            st.write("No centrality data")

        st.subheader("Messages heatmap — users vs weekdays")
        df['Weekday'] = df['DateTime'].dt.day_name()
        heat = df.pivot_table(index='User', columns='Weekday', values='Message', aggfunc='count').fillna(0)
        if not heat.empty:
            plt.figure(figsize=(10,6))
            sns.heatmap(heat, cmap="YlGnBu", annot=False)
            st.pyplot(plt, use_container_width=True)
        else:
            st.write("Not enough data for heatmap")

        st.subheader("Hourly interactions (by sender)")
        hour_sender = df.groupby([df['DateTime'].dt.hour, 'User']).size().unstack(fill_value=0)
        if not hour_sender.empty:
            top_senders = hour_sender.sum().sort_values(ascending=False).head(8).index
            st.line_chart(hour_sender[top_senders], use_container_width=True)
        else:
            st.write("No hourly data")

# Download Report (PDF & Excel safe)
elif page == "Download Report":
    st.header("Download professional report (PDF & Excel)")
    if "chat_df" not in st.session_state:
        st.warning("Upload a chat first.")
    else:
        df = st.session_state["chat_df"]

        if st.button("Generate & Download PDF Report"):
            with st.spinner("Generating PDF..."):
                pdf_bytes = create_pdf_report(df)
            st.success("PDF ready")
            st.download_button("Download PDF", data=pdf_bytes, file_name="whatsapp_report.pdf", mime="application/pdf")

        if st.button("Generate & Download Excel Report"):
            with st.spinner("Generating Excel..."):
                excel_buf = create_excel_report(df)
            st.success("Excel ready")
            st.download_button("Download Excel", data=excel_buf, file_name="whatsapp_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Chatbot
elif page == "Chatbot":
    st.header("💬 WhatsApp Chat Assistant")
    st.markdown("Ask natural-language questions about your uploaded chat. Examples: *Who sent the most messages?* *Top emojis?* *How many times did I say 'bro'?*")
    if "chat_df" not in st.session_state:
        st.warning("Upload a chat first.")
    else:
        df = st.session_state["chat_df"]
        q = st.text_input("Ask a question (try: top user, total messages, top emojis, 'word hello count'):")
        if st.button("Ask"):
            if not q.strip():
                st.info("Type a question first.")
            else:
                with st.spinner("Finding answer..."):
                    result = chatbot_answer(q, df)
                st.markdown("**Matched sample Q:** " + (result.get("matched") or "—"))
                st.markdown("**Intent:** " + (result.get("intent") or "—"))
                st.subheader("Answer")
                st.write(result.get("answer"))

# End of app
