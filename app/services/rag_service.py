# 1. 라이브러리 임포트
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.schemas.chat import ChatRequest
from .dummy_data import PRODUCT_DOCUMENTS

# 2. 전역 변수 설정 (단 한 번만 실행되도록)
VECTOR_STORE = None


# 3. 데이터 로딩 및 벡터 스토어 구축 함수
def initialize_vector_store():
    global VECTOR_STORE
    if VECTOR_STORE is not None:
        return

    # a. 텍스트 데이터를 문서 객체로 변환
    # 실제로는 `UnstructuredFileLoader` 등을 사용해 파일을 불러옵니다.
    # 예시를 위해 단순 텍스트를 문서 객체로 변환합니다.
    documents = [
        {"page_content": doc, "metadata": {"source": "product"}}
        for doc in PRODUCT_DOCUMENTS
    ]

    # b. 텍스트를 적절한 크기로 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.create_documents([d["page_content"] for d in documents])

    # c. 임베딩 모델 선택 및 초기화
    # 한국어에 잘 동작하는 모델을 사용하거나, 일반적인 multilingual 모델을 사용합니다.
    # 여기서는 범용적인 모델을 사용합니다.
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # d. ChromaDB에 문서 임베딩 및 저장 (in-memory로 간단히 테스트)
    VECTOR_STORE = Chroma.from_documents(docs, embedding_model)
    print("✅ ChromaDB vector store initialized!")


# 4. 사용자 쿼리를 받아 검색하는 함수
def get_retrieved_documents(request: ChatRequest) -> list:
    """사용자의 쿼리를 받아 벡터 DB에서 관련 문서를 검색합니다."""

    # 벡터 스토어에서 쿼리와 가장 유사한 문서 3개를 검색합니다.
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(request.query)

    # 검색된 문서의 내용을 리스트로 반환
    return [doc.page_content for doc in relevant_docs]
