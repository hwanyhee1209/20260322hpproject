# Samsung Card Manual QA Bot

특정 문서(삼성전자 메모리카드 매뉴얼)를 기반으로 사용자 질문에 정확한 정보를 찾아 답변해주는 **RAG(Retrieval-Augmented Generation)** 기반 챗봇 서비스이다

## 프로젝트 소개
이 프로젝트는 **LangChain**과 **OpenAI**를 활용하여 PDF 문서 내의 지식을 추출하고, **Streamlit**을 통해 웹 인터페이스로 제공합니다. 사용자가 매뉴얼의 긴 내용을 직접 읽지 않아도 질문을 통해 필요한 정보를 즉시 얻을 수 있다

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
```

### 2. 가상환경 생성 및 활성화 그리고 필요 라이브러리설치
```bash
python -m venv venv
source venv/scripts/activate  # Windows 환경
pip install -r requirements.txt
```
### 3. 환경 변수 설정

.env 파일을 루트 디렉토리에 생성하고 OpenAI API 키를 입력한다
```bash
OPENAI_API_KEY=your_api_key_here
```

### 4. 실행
```bash
streamlit run app.py
```
