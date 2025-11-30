import streamlit as st
import pandas as pd
import plotly.express as px
import os
import zipfile
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Auto-unzip faiss_index if not already done
if not os.path.exists("faiss_index"):
    with zipfile.ZipFile("faiss_index.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge – AI Business Intelligence Assistant")

df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
        memory=memory
    )

qa = load_chain()

with st.sidebar:
    st.header("Quick Charts")
    if st.button("Monthly Trend"):
        m = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        m['Date'] = m['Date'].dt.to_timestamp()
        st.plotly_chart(px.line(m, x='Date', y='Sales'), use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about your sales data!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans = qa.invoke({"question": prompt})["answer"]
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
