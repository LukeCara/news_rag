import streamlit as st
import os
from utils.embedding_utils import EmbeddingModel
from utils.vector_store import VectorStore
from utils.llm_utils import LLMHandler
from utils.schema import Question, Response, Document
from utils.document_processor import DocumentProcessor
import json
import numpy as np

# Load custom CSS
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
# Initialize session state
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        try:
            model_path = os.environ.get("BGE_MODEL_PATH", "BAAI/bge-m3")
            st.session_state.embedding_model = EmbeddingModel(model_path)
            st.session_state.vector_store = VectorStore(
                dimension=st.session_state.embedding_model.get_embedding_dimension()
            )
            st.session_state.document_processor = DocumentProcessor()
        except Exception as e:
            st.error(f"初始化嵌入模型时出错：{str(e)}")
            st.session_state.embedding_model = None
            st.session_state.vector_store = None
            st.session_state.document_processor = None
            
    if 'llm_handler' not in st.session_state:
        try:
            model_path = os.environ.get("LLM_MODEL_PATH", "/media/mynewdrive/llm_store/model_store/qwen2-7b-instruct")
            if not os.path.exists(model_path):
                st.warning(f"模型路径未找到：{model_path}")
                st.info("请设置 LLM_MODEL_PATH 环境变量指向您的 Qwen 模型。")
                st.session_state.llm_handler = None
            else:
                st.session_state.llm_handler = LLMHandler(model_path)
        except Exception as e:
            st.error(f"初始化 LLM 时出错：{str(e)}")
            st.session_state.llm_handler = None

def display_chat_history():
    """Display the chat history."""
    for message in st.session_state.chat_history:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"**您：** {message['content']}")
            else:
                st.markdown(f"**助手：** {message['content']}")
                if message.get("sources"):
                    with st.expander("参考来源"):
                        for source in message["sources"]:
                            st.write(source)

def process_question(question: str):
    """Process a user question and generate a response."""
    try:
        if not st.session_state.llm_handler:
            return Response(
                answer="抱歉，语言模型未正确初始化。请确保模型路径设置正确。",
                confidence=0.0
            )
        
        question_embedding = st.session_state.embedding_model.generate_embeddings(question)
        search_results = st.session_state.vector_store.search(
            question_embedding,
            k=3
        )
        
        context = []
        for doc, score in search_results:
            context.append(f"[来源：{doc.metadata['filename']}, 片段：{doc.metadata['chunk_id'] + 1}/{doc.metadata['total_chunks']}]\n{doc.content}")
        
        recent_history = st.session_state.chat_history[-8:] if len(st.session_state.chat_history) > 0 else None
        
        response_placeholder = st.empty()
        response_text = ""
        
        for partial_response in st.session_state.llm_handler.generate_response_stream(
            question=question,
            context=context,
            chat_history=recent_history
        ):
            response_text += partial_response
            response_placeholder.markdown(f"**助手：** {response_text}")
        
        return response_text
        
    except Exception as e:
        st.error(f"处理问题时出错：{str(e)}")
        return "抱歉，处理您的问题时遇到错误。请重试。"

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and add it to the vector store."""
    try:
        documents = st.session_state.document_processor.process_file(
            uploaded_file,
            uploaded_file.name
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doc in enumerate(documents):
            progress = (i + 1) / len(documents)
            progress_bar.progress(progress)
            status_text.text(f"正在处理第 {i + 1}/{len(documents)} 个片段...")
            
            embeddings = st.session_state.embedding_model.generate_embeddings(doc.content)
            st.session_state.vector_store.add_documents([doc], embeddings.reshape(1, -1))
        
        progress_bar.empty()
        status_text.empty()
        
        return len(documents)
        
    except Exception as e:
        raise Exception(f"处理文件时出错：{str(e)}")

def main():
    st.set_page_config(
        page_title="AIR",
        page_icon="🤖",
        layout="wide"
    )
    
    load_css()
    init_session_state()
    st.title("AIR, 你的产业情报助手")
    
    with st.sidebar:
        st.header("设置")
        
        uploaded_files = st.file_uploader(
            "上传文档",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    with st.spinner(f"正在处理 {uploaded_file.name}..."):
                        num_chunks = process_uploaded_file(uploaded_file)
                        st.success(f"成功将 {uploaded_file.name} 处理为 {num_chunks} 个片段！")
                        
                except Exception as e:
                    st.error(f"处理 {uploaded_file.name} 时出错：{str(e)}")
    
    display_chat_history()
    
    question = st.text_input("请输入您的问题：")
    
    if st.button("发送"):
        if question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            response = process_question(question)
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            st.session_state.question = ""
            st.rerun()
        else:
            st.warning("请输入问题。")

if __name__ == "__main__":
    main()
