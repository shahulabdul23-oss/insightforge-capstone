# app.py — 100% WORKING VERSION FOR STREAMLIT CLOUD
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge – AI Business Intelligence Assistant")

# Load data
df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Load vector store
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

# Sidebar charts
with st.sidebar:
    st.header("Quick Insights")
    if st.button("Monthly Sales Trend"):
        monthly = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly['Date'] = monthly['Date'].dt.to_timestamp()
        st.plotly_chart(px.line(monthly, x='Date', y='Sales', title="Sales Trend"), use_container_width=True)
    if st.button("Top Products"):
        top = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10)
        st.plotly_chart(px.bar(top, color=top.values, color_continuous_scale="Viridis"), use_container_width=True)

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm InsightForge. Ask me anything about your sales data!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about products, regions, trends..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = qa.invoke({"question": prompt})["answer"]
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
