import streamlit as st
import pandas as pd, plotly.express as px, os
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge – AI Business Intelligence Assistant")

df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

@st.cache_resource
def get_qa():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vs.as_retriever(), memory=memory)

qa = get_qa()

with st.sidebar:
    st.header("Quick Charts")
    if st.button("Monthly Trend"):
        m = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        m['Date'] = m['Date'].dt.to_timestamp()
        st.plotly_chart(px.line(m, x='Date', y='Sales'), use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about the sales data."}]

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if q := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": q})
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans = qa.invoke({"question": q})["answer"]
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})

