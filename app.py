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

# Auto-unzip faiss_index
if not os.path.exists("faiss_index") and os.path.exists("faiss_index.zip"):
    with zipfile.ZipFile("faiss_index.zip", 'r') as z:
        z.extractall(".")

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge – AI Business Intelligence Assistant")

df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    #  reads key from Secrets
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY") 
    )
    
    memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

qa = load_chain()

# l UI
with st.sidebar:
    st.header("Quick Charts")
    if st.button("Monthly Sales Trend"):
        m = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        m['Date'] = m['Date'].dt.to_timestamp()
        st.plotly_chart(px.line(m, x='Date', y='Sales', title="Monthly Trend"), use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm InsightForge. Ask me anything about the sales data!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about sales, products, regions..."):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            ans = qa.invoke({"question": prompt})["answer"]
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
