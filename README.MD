# Samsung Card Manual QA Bot

특정 문서(삼성전자 메모리카드 매뉴얼)를 기반으로 사용자 질문에 정확한 정보를 찾아 답변해주는 **RAG(Retrieval-Augmented Generation)** 기반 챗봇 서비스입니다.

## 프로젝트 소개
이 프로젝트는 **LangChain**과 **OpenAI**를 활용하여 PDF 문서 내의 지식을 추출하고, **Streamlit**을 통해 웹 인터페이스로 제공합니다. 사용자가 매뉴얼의 긴 내용을 직접 읽지 않아도 질문을 통해 필요한 정보를 즉시 얻을 수 있습니다.

##  기술 스택
- **Language**: Python 3.10
- **Framework**: Streamlit
- **LLM Orchestration**: LangChain
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Model**: OpenAI GPT-4o-mini / text-embedding-3-small
- **Library**: PyPDF, python-dotenv, tiktoken

## 설치 및 실행 방법

### 1. 저장소 클론
```bash
git clone [https://github.com/사용자계정/프로젝트명.git](https://github.com/사용자계정/프로젝트명.git)
cd 프로젝트명