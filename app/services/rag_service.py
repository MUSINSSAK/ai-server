# 1. 라이브러리 임포트
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader

# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.schemas.chat import ChatRequest
from dotenv import load_dotenv

# OpenAI LLM, 프롬프트 템플릿, 체인을 위한 라이브러리
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate  # <--- 이 부분이 추가되었습니다.
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

VECTOR_STORE = None
RAG_CHAIN = None


# 3. 데이터 로딩 및 벡터 스토어 구축 함수
def initialize_vector_store():
    global VECTOR_STORE, RAG_CHAIN
    if VECTOR_STORE is not None and RAG_CHAIN is not None:
        return

    # a. CSVLoader를 사용해 파일을 불러옵니다.
    # file_path: 데이터 파일 경로
    # source_column: 검색 결과 메타데이터로 사용할 열 이름
    loader = CSVLoader(
        file_path="./data/products.csv", source_column="product_id", encoding="utf-8"
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents from CSV.")

    # b. 텍스트를 적절한 크기로 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(docs)} chunks.")

    # c. 임베딩 모델 선택 및 초기화
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # d. ChromaDB에 문서 임베딩 및 저장
    VECTOR_STORE = Chroma.from_documents(docs, embedding_model)
    print("✅ ChromaDB vector store initialized!")

    # 4. 프롬프트 템플릿 정의 및 RAG 체인 구축 (이전과 동일)
    # ... 이 부분은 이전 단계의 코드를 그대로 유지합니다.
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
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
    document_chain = create_stuff_documents_chain(llm, prompt)
    RAG_CHAIN = create_retrieval_chain(retriever, document_chain)


def get_rag_answer(request: ChatRequest) -> str:
    response = RAG_CHAIN.invoke({"input": request.query})
    return response["answer"]
