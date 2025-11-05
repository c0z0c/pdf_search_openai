"""
VectorStore 사용 예제

PDF 문서를 벡터스토어에 추가하고 검색하는 기본 예제입니다.
"""

from pathlib import Path
from langchain_openai import ChatOpenAI
import sys

# src 폴더를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pdf_search import VectorStore

def main():
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    
    # VectorStore 생성
    db_path = str(data_path / 'vectorstore')
    vector_store = VectorStore(
        llm=llm,
        chunk_size=600,
        chunk_overlap=100,
        db_path=db_path
    )
    
    # 1. PDF 문서 추가
    print("="*60)
    print("1. PDF 문서 추가")
    print("="*60)
    
    pdf_files = [
        # 여기에 PDF 파일 경로를 추가하세요
        # str(data_path / 'sample.pdf'),
    ]
    
    if pdf_files and all(Path(f).exists() for f in pdf_files):
        vector_store.add_documents(pdf_files)
        vector_store.save("my_knowledge_base")
        print(f"\n{len(pdf_files)}개 문서 추가 완료\n")
    else:
        print("\nPDF 파일을 찾을 수 없습니다. 기존 벡터스토어를 로드합니다.\n")
        try:
            vector_store.load("my_knowledge_base")
        except FileNotFoundError:
            print("기존 벡터스토어도 없습니다. 문서를 먼저 추가해주세요.")
            return
    
    # 2. 메타데이터 조회
    print("="*60)
    print("2. 벡터스토어 메타데이터")
    print("="*60)
    metadata_info = vector_store.get_metadata_info()
    print(metadata_info)
    print()
    
    # 3. 검색 및 답변 생성
    print("="*60)
    print("3. 검색 및 답변 생성")
    print("="*60)
    
    queries = [
        "RAG의 핵심 원리는 무엇인가요?",
        # 추가 질문들을 여기에 작성하세요
    ]
    
    for query in queries:
        print(f"\n질의: {query}")
        print("-"*60)
        
        # 검색 결과
        results = vector_store.search(query)
        print(f"\n검색 결과: {len(results)}개")
        for idx, result in enumerate(results, 1):
            print(f"  [{idx}] {result['file_name']} (p.{result['page']}) - 유사도: {result['score']:.3f}")
        
        # RAG 답변
        answer = vector_store.generate_answer(query)
        print(f"\n답변:\n{answer}")
        print()

if __name__ == "__main__":
    main()
