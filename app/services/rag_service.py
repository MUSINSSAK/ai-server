# 1. 라이브러리 임포트
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.schemas.chat import ChatRequest
from .dummy_data import PRODUCT_DOCUMENTS

# OpenAI LLM, 프롬프트 템플릿, 체인을 위한 라이브러리 추가
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv  # .env 파일 로드를 위해 추가

# .env 파일에서 환경 변수 로드
load_dotenv()

# 2. 전역 변수 설정
VECTOR_STORE = None
RAG_CHAIN = None


# 3. 데이터 로딩 및 벡터 스토어 구축 함수
def initialize_vector_store():
    global VECTOR_STORE, RAG_CHAIN
    if VECTOR_STORE is not None and RAG_CHAIN is not None:
        return

    # ChromaDB를 위한 문서 준비 (이전과 동일)
    documents = [
        {"page_content": doc, "metadata": {"source": "product"}}
        for doc in PRODUCT_DOCUMENTS
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([d["page_content"] for d in documents])
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    VECTOR_STORE = Chroma.from_documents(docs, embedding_model)
    print("✅ ChromaDB vector store initialized!")

    # 4. 프롬프트 템플릿 정의
    SYSTEM_PROMPT = """
    You are a helpful assistant for an e-commerce platform.
    Your task is to answer the user's question based on the provided product information.
    If the information is not relevant to the question, state that you cannot provide an answer based on the given context.
    Respond in a friendly and professional tone.
    Do not use information from outside the provided context.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    # 5. RAG 체인 구축
    # a. LLM 초기화 (비용 절감을 위해 gpt-3.5-turbo 사용)
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # b. Retriever 생성
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})

    # c. 체인 정의: 검색된 문서를 프롬프트에 넣고 LLM을 호출하는 체인
    document_chain = create_stuff_documents_chain(llm, prompt)
    RAG_CHAIN = create_retrieval_chain(retriever, document_chain)


# 6. 사용자 쿼리를 받아 RAG 파이프라인을 실행하는 함수
def get_rag_answer(request: ChatRequest) -> str:
    """
    사용자의 쿼리를 받아 RAG 파이프라인을 실행하고 최종 답변을 반환합니다.
    """
    # 전역 변수에서 체인 실행
    response = RAG_CHAIN.invoke({"input": request.query})

    return response["answer"]
