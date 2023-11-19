#!/usr/bin/env python
# coding: utf-8


import os
import time
import json
import sys
from typing import Any, Iterable, List
from PyPDF2 import PdfReader
import langchain
from langchain.docstore.document import Document

import openai

import pandas as pd

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")
sys.path.append(os.getenv("PYTHONPATH"))

from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.schema.vectorstore import (
  VectorStore,
  VectorStoreRetriever
)
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from utils import (
  BusyIndicator,
  ConsoleInput,
  load_pdf,
  load_pdf_vectordb,
  load_vectordb_from_file
  )
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chains.summarize import load_summarize_chain
import re
from langchain.document_loaders import PyPDFLoader



def reduce_newlines(input_string):
  # 정규 표현식을 사용하여 연속된 '\n'을 하나로 치환
  reduced_string = re.sub(r'\n{3,}', '\n\n', input_string)
  return reduced_string


def print_documents(docs: List[Any]) -> None:
  if docs == None:
    return

  print(f"documents size: {len(docs)}")
  p = lambda meta, key: print(f"{key}: {meta[key]}") if key in meta else None
  for doc in docs:
    print(f"source : {doc.metadata['source']}")
    p(doc.metadata, 'row')
    p(doc.metadata, 'page')
    print(f"content: {reduce_newlines(doc.page_content)[0:500]}")
    print('-'*30)

def print_result(result: Any) -> None:
  p = lambda key: print(f"{key}: {result[key]}") if key in result else None
  p('query')
  p('question')
  print(f"result: {'-' * 22}" )
  p('result')
  p('answer')
  print('-'*30)
  if 'source_documents' in result:
    print("documents")
    print_documents(result['source_documents'])
    

# 총 페이지 수를 계산하는 함수
def get_total_page_count(pdf_file_path) -> int:
    reader = PdfReader(pdf_file_path)
    return len(reader.pages)
  
# 계산된 페이지 수를 문서화
def get_page_count_document() -> Document:
    page_count = get_total_page_count(PDF_FREELANCER_GUIDELINES_FILE)
    content = f"The total page count of the document is {page_count}."
    metadata = {"description": "Page count information"}

    # `type` 속성을 "Document"로 고정
    return Document(page_content=content, metadata=metadata, type="Document")


llm_model = "gpt-3.5-turbo-1106"
PDF_FREELANCER_GUIDELINES_FILE = "./data/프리랜서 가이드라인(출판본).pdf"
CSV_OUTDOOR_CLOTHING_CATALOG_FILE = "./data/OutdoorClothingCatalog_1000.csv"

llm = ChatOpenAI(model_name=llm_model, temperature=0)

#목차 추출
def get_table_of_contents(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load()

    table_of_contents = []
    for page in pages:
        text = page.page_content
        # 목차를 식별하고 추출하는 로직 구현
        # 예시: 특정 키워드나 패턴을 찾아 목차 식별
        if "목차" in text or "CHAPTER" in text:
            table_of_contents.append(text)

    return table_of_contents
  
#
def infer_gender(product_name, product_description):
    female_keywords = ['women', 'woman', 'female', 'girls', 'girl']
    male_keywords = ['men', 'man', 'male', 'boys', 'boy']
    name_desc = product_name.lower() + " " + product_description.lower()
    if any(keyword in name_desc for keyword in female_keywords):
        return 'Female'
    elif any(keyword in name_desc for keyword in male_keywords):
        return 'Male'
    else:
        return 'Unisex'

def infer_season(product_name, product_description):
    summer_keywords = ['summer', 'hot', 'warm', 'sun']
    winter_keywords = ['winter', 'cold', 'snow', 'ice']
    all_season_keywords = ['all season', 'any season', 'all-weather']
    name_desc = product_name.lower() + " " + product_description.lower()
    if any(keyword in name_desc for keyword in summer_keywords):
        return 'Summer'
    elif any(keyword in name_desc for keyword in winter_keywords):
        return 'Winter'
    elif any(keyword in name_desc for keyword in all_season_keywords):
        return 'All Season'
    else:
        return 'General'

# 
def calculate_product_statistics(catalog_df):
    # 성별과 계절을 추론하는 로직 적용
    catalog_df['Inferred Gender'] = catalog_df.apply(lambda row: infer_gender(row['name'], row['description']), axis=1)
    catalog_df['Inferred Season'] = catalog_df.apply(lambda row: infer_season(row['name'], row['description']), axis=1)

    # 통계 계산
    gender_counts = catalog_df['Inferred Gender'].value_counts().to_dict()
    season_counts = catalog_df['Inferred Season'].value_counts().to_dict()

    return gender_counts, season_counts
  


def get_personal_retriever() -> VectorStoreRetriever:
  personal_texts = [
    "내 이름은 안랩샘아카데미17기 연습용 챗봇입니다.",
    "내가 제일 좋아하는 색은 연초록색입니다.",
    "내 꿈은 안랩샘아카데미17기 연습용 챗봇으로 여러분에게 도움을 되는 것입니다.",
    "나는 2023년 안랩샘아카데미17기 ChatGPT활용 챗봇 개발교육 과정에서 탄생했습니다.",
    "나는 현재 .PDF파일과 .csv파일 두 가지 타입의 데이터를 학습하여 답변을 하고 있습니다.",
  ]
  personal_retriever = FAISS.from_texts(personal_texts, OpenAIEmbeddings()).as_retriever()
  if not isinstance(personal_retriever, VectorStoreRetriever):
    raise ValueError("personal_retriever is not VectorStoreRetriever")
  return personal_retriever

def get_freelancer_guidelines_summary() -> VectorStoreRetriever:
    # Load the freelancer guidelines document
    docs = load_pdf(PDF_FREELANCER_GUIDELINES_FILE)
    
    # Summarize the content using TextRank algorithm
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    
    # Create a VectorStoreRetriever and add the summarized content as a document
    retriever = VectorStoreRetriever()
    document = Document("Summary", summary)
    retriever.add_document(document)
    
    return retriever
  
def get_freelancer_guidelines() -> VectorStoreRetriever:
  retriever = load_vectordb_from_file(PDF_FREELANCER_GUIDELINES_FILE).as_retriever()
  if not isinstance(retriever, VectorStoreRetriever):
    raise ValueError("it's not VectorStoreRetriever")
  return retriever

def get_freelancer_guidelines_summary() -> VectorStoreRetriever:
  retriever = load_vectordb_from_file(PDF_FREELANCER_GUIDELINES_FILE).as_retriever()
  if not isinstance(retriever, VectorStoreRetriever):
    raise ValueError("it's not VectorStoreRetriever")
  return retriever

# 총 페이지 계산
def get_page_count_retriever() -> VectorStoreRetriever:
    page_count_doc = get_page_count_document()
    embedding = OpenAIEmbeddings()

    # 'page_content' 속성을 사용하여 텍스트를 가져옵니다
    vectorstore = FAISS.from_texts([page_count_doc.page_content], embedding)
    vectorstore.add_texts([page_count_doc.page_content])

    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_table_of_contents_retriever() -> VectorStoreRetriever:
    table_of_contents = get_table_of_contents(PDF_FREELANCER_GUIDELINES_FILE)

    # 임베딩 함수 준비
    embedding = OpenAIEmbeddings()

    # 임베딩 함수와 함께 FAISS 인스턴스 생성
    contents = [item for item in table_of_contents]
    vectorstore = FAISS.from_texts(contents, embedding)

    # 추출된 텍스트를 vectorstore에 추가
    for item in table_of_contents:
        vectorstore.add_texts([item])

    # 채워진 vectorstore를 사용하여 VectorStoreRetriever 인스턴스 생성
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_outdoor_clothing_catalog() -> VectorStoreRetriever:
  retriever = load_vectordb_from_file(CSV_OUTDOOR_CLOTHING_CATALOG_FILE).as_retriever()
  if not isinstance(retriever, VectorStoreRetriever):
    raise ValueError("it's not VectorStoreRetriever")
  return retriever

# OutdoorClothingCatalog_1000.csv 데이터를 분석하여 상품 정보 개괄 및 통계 정보를 제공하는 함수
def get_outdoor_clothing_stats() -> VectorStoreRetriever:
    catalog_df = pd.read_csv(CSV_OUTDOOR_CLOTHING_CATALOG_FILE)

    # 통계 정보 계산
    total_products = len(catalog_df)
    gender_counts, season_counts = calculate_product_statistics(catalog_df)

    # 통계 정보를 문자열로 변환하여 상품 정보 개괄 및 통계 정보 생성
    stats_info = f"안녕하세요. 아웃도어 전문 매장입니다. 현재 저희는 {total_products}개의 제품을 다루고 있습니다. 성별 의류 수: {gender_counts}, 계절별 의류 수: {season_counts}."

    # 임베딩 및 VectorStoreRetriever 생성
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts([stats_info], embedding)
    vectorstore.add_texts([stats_info])
    retriever = VectorStoreRetriever(vectorstore=vectorstore)

    return retriever




def get_tools() :
  tools = [
    create_retriever_tool(
      get_freelancer_guidelines(),
      "freelancer_guidelines",
      "Good for answering questions about the different things you need to know about being a freelancer",
    ),
    create_retriever_tool(
      get_freelancer_guidelines_summary(),
      "freelancer_guidelines",
      "Good for answering questions about summarize the full contents of your freelance guidelines",
    ),
    create_retriever_tool(
            get_page_count_retriever(),
            "page_count",
            "Provides the total page count of the Freelancer Guidelines document"
    ),
    create_retriever_tool(
        get_table_of_contents_retriever(),
        "table_of_contents",
        "Provides the table of contents of the Freelancer Guidelines document"
    ),
    # 새로운 tool 정의 추가
    create_retriever_tool(
        get_outdoor_clothing_stats(),
        "outdoor_clothing_stats",
        "Provides an overview and statistical information of the outdoor clothing catalog"
    ),
    create_retriever_tool(
      get_outdoor_clothing_catalog(),
      "outdoor_clothing_catalog",
      "Good for answering questions about outdoor clothing names and features",
    ),
    create_retriever_tool(
      get_personal_retriever(),
      "personal",
      "Good for answering questions about me",
    )
  ]
  return tools




def chat_qa(is_debug=False) -> None:
  console = ConsoleInput(basic_prompt='% ')
  busy_indicator = BusyIndicator().busy(True, "vectordb를 로딩중입니다 ")
  tools = get_tools()
  busy_indicator.stop()

  agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

  while True:  # 무한루프 시작
    t = console.input()[0].strip()

    if t == '':  # 빈 라인인 경우.
      continue

    if t == 'q' or t == 'Q' or t == 'ㅂ':
      break

    busy_indicator = BusyIndicator().busy(True)
    langchain.is_debug = is_debug
    result = agent_executor({"input": t})
    langchain.is_debug = False
    busy_indicator.stop()
    console.out(result["output"])
    if is_debug:
      print_result(result)

def input_select(menu: dict) -> (int, str):
  print(menu.get("title"))
  items = menu.get("items", None)

  if items == None or len(items) == 0:
     raise ValueError("menu에 items가 없습니다.")
  for idx, item in enumerate(items):
    print(f"{str(idx+1)}. {item}")

  size = len(items)
  select = -1

  while select < 0:
    try:
      select = int(input(">>선택 :"))
      if select <= 0 or size < select:
        select = -1
    except ValueError:
      select = -1

    if select < 0:
      print("잘못된 선택입니다.")

  return ( select, items[select-1] )




def main():
  debug, _ = input_select({
    "title" : "debugging 모드로 하시겠습니까?",
    "items" : [
      "yes",
      "no"
    ]
  })

  is_debug = debug == 1

  chat_qa( is_debug)



if __name__ == '__main__':
  main()
