

### 여기에 정답을 작성해보세요

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter# 문서 청크 분할
from langchain_community.vectorstores import FAISS# 벡터DB로 FAISS 사용
from langchain_openai import OpenAIEmbeddings, ChatOpenAI# OpenAI 임베딩/LLM
from langchain_core.prompts import ChatPromptTemplate# RAG 프롬프트
from langchain_core.runnables import RunnablePassthrough## 사용자가 입력한 질문이 그대로 prompt의 변수로 넣기
from langchain_core.output_parsers import StrOutputParser  # 출력 파서


# 1. 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="Samsung Card Manual Bot", page_icon="💾")

# 캐싱을 통해 매번 PDF를 다시 로드하고 임베딩하지 않도록 설정
@st.cache_resource
def prepare_rag_chain():
    # PDF 로드 (경로 주의: data 폴더 안에 파일이 있어야 함)
    pdf_path = "data/Samsung_Card_Manual_Korean_1.3.pdf"
    if not os.path.exists(pdf_path):
        return None
        
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # 벡터 DB 및 리트리버 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_template("""
    너는 삼성전자 메모리카드 매뉴얼에 대한 전문 어시스턴트이다.
    다음의 참고 문서를 바탕으로 질문에 정확하게 답하라.

    [참고문서]
    {context}

    [질문]
    {question}

    한글로 간결하고 정확하게 답변하라.
    """)

    # RAG 체인 구성 (모델명은 실제 사용 가능한 gpt-4o-mini 등으로 변경 권장)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# --- UI 레이아웃 ---
st.title("삼성 메모리카드 매뉴얼 챗봇")
st.markdown("삼성전자 메모리카드 유틸리티 매뉴얼을 기반으로 답변하는 챗봇입니다.")

# 체인 초기화
rag_chain = prepare_rag_chain()

if rag_chain is None:
    st.error("PDF 파일을 찾을 수 없습니다. 'data/' 폴더에 매뉴얼 파일을 넣어주세요.")
else:
    # 대화 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 메시지 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("매뉴얼에 대해 궁금한 점을 물어보세요."):
        # 사용자 메시지 저장 및 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 챗봇 응답 생성 및 표시
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성 중입니다..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
        
        # 응답 저장
        st.session_state.messages.append({"role": "assistant", "content": response})
     