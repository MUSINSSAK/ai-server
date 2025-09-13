import os
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.schemas.chat import ChatRequest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

# 전역 변수 설정
VECTOR_STORE = None
RAG_CHAIN = None


def initialize_vector_store():
    global VECTOR_STORE, RAG_CHAIN
    if VECTOR_STORE is not None and RAG_CHAIN is not None:
        return

    PERSIST_DIRECTORY = "./chroma_db"  # <-- 영속성 폴더 이름 추가

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    if os.path.exists(PERSIST_DIRECTORY):
        # 파일이 존재하면, 기존 벡터 스토어를 불러오기
        print("✅ 기존 벡터 스토어를 불러오는 중...")
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model
        )
    else:
        # 파일이 없으면, 새로 생성하고 저장하기
        print("✅ 기존 벡터 스토어가 없어 새로 생성하는 중...")

        loader = CSVLoader(
            file_path="./data/products.csv",
            source_column="product_id",
            encoding="utf-8",
        )
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents from CSV.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        print(f"✅ Split into {len(docs)} chunks.")

        vector_store = Chroma(
            embedding_function=embedding_model, persist_directory=PERSIST_DIRECTORY
        )  # <-- persist_directory 추가

        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            vector_store.add_documents(batch)
            print(f"  - {i+len(batch)}/{len(docs)} 청크 처리 완료...")
        print("✅ 모든 청크를 배치로 나누어 벡터 스토어에 추가했습니다.")

    VECTOR_STORE = vector_store  # 전역 변수에 할당

    # RAG 체인 구축 (이전과 동일)
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
    llm = ChatOpenAI(model="gpt-4o")
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
    document_chain = create_stuff_documents_chain(llm, prompt)
    RAG_CHAIN = create_retrieval_chain(retriever, document_chain)


def get_rag_answer(request: ChatRequest) -> dict:  # <-- 반환 타입을 dict로 변경
    """
    사용자의 쿼리를 받아 RAG 파이프라인을 실행하고 최종 답변과 ID를 반환합니다.
    """
    response = RAG_CHAIN.invoke({"input": request.query})

    # 검색된 문서에서 product_id(source)를 추출합니다.
    retrieved_ids = [doc.metadata["source"] for doc in response["context"]]

    # 중복을 제거하여 유니크한 ID만 남깁니다.
    unique_ids = list(dict.fromkeys(retrieved_ids))

    return {"answer": response["answer"], "recommended_ids": unique_ids}
