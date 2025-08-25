import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()

VECTOR_STORE = None
RAG_CHAIN = None


def initialize_rag_components():
    """RAG 파이프라인에 필요한 모든 컴포넌트를 초기화하는 함수"""
    global VECTOR_STORE, RAG_CHAIN

    PERSIST_DIRECTORY = "./chroma_db"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    if os.path.exists(PERSIST_DIRECTORY):
        print("✅ 기존 벡터 스토어를 불러오는 중...")
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model
        )
    else:
        print("✅ 기존 벡터 스토어가 없어 새로 생성하는 중...")
        loader = CSVLoader(
            file_path="./data/products.csv",
            source_column="product_id",
            encoding="utf-8",
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        vector_store = Chroma(
            embedding_function=embedding_model, persist_directory=PERSIST_DIRECTORY
        )
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            vector_store.add_documents(batch)
            print(f"  - {i+len(batch)}/{len(docs)} 청크 처리 완료...")

    VECTOR_STORE = vector_store
    print("✅ 벡터 스토어 초기화 완료!")

    SYSTEM_PROMPT = """
    You are a helpful assistant for an e-commerce platform.
    Your task is to answer the user's question based on the provided product information.
    If there is a product related to the user's question, please clearly mention the brand and product name of that product.
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
    print("✅ RAG 체인 초기화 완료!")


def test_full_rag_pipeline(query: str):
    """주어진 쿼리에 대해 RAG의 전체 파이프라인을 테스트하는 함수입니다."""
    print(f"\n\n--- 전체 RAG 파이프라인 테스트: '{query}' ---")
    if RAG_CHAIN is None:
        print(
            "❌ RAG 체인이 초기화되지 않았습니다. initialize_rag_components()를 먼저 호출해주세요."
        )
        return

    print("🔎 검색 및 답변 생성 시작...")
    response = RAG_CHAIN.invoke({"input": query})

    print("\n--- LLM 최종 답변 ---")
    print(response["answer"])
    print("-" * 50)


def test_retrieval_only(query: str):
    """주어진 쿼리에 대해 검색(Retrieval) 단계만 테스트하는 함수입니다."""
    print(f"\n\n--- 검색 전용 테스트: '{query}' ---")
    if VECTOR_STORE is None:
        print(
            "❌ 벡터 스토어가 초기화되지 않았습니다. initialize_rag_components()를 먼저 호출해주세요."
        )
        return

    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)

    print("🔎 검색 결과:")
    for i, doc in enumerate(relevant_docs):
        print(f"📄 [문서 {i+1}]")
        print(f"내용: {doc.page_content}")
        if "source" in doc.metadata:
            print(f"출처(product_id): {doc.metadata['source']}")
        print("-" * 50)


if __name__ == "__main__":
    initialize_rag_components()  # 모든 컴포넌트 초기화

    # 테스트하고 싶은 질문들
    test_query_1 = "마크니 브랜드의 가방에 대한 리뷰를 요약해줘."
    test_query_2 = "키링이 포함된 상품 있어?"
    test_query_3 = "여행갈 때 들기 좋은 가방 추천해줘."
    test_query_4 = "빨간색 신발 추천해줘."  # 데이터에 없는 질문
    test_query_5 = "키가 큰 여자친구한테 선물로 사 줄 가방 추천좀"
    test_query_6 = "쿨톤에 맞는 아우터가 있을까?"
    test_query_7 = "해외여행갈때 쓸 가방 추천좀 해주라"

    # 1. 검색 전용 테스트
    test_retrieval_only(test_query_7)

    # 2. 전체 RAG 파이프라인 테스트
    # test_full_rag_pipeline(test_query_1)
    # test_full_rag_pipeline(test_query_2)
    # test_full_rag_pipeline(test_query_3)
    # test_full_rag_pipeline(test_query_4)
    test_full_rag_pipeline(test_query_7)
