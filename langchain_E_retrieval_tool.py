#!/usr/bin/env python
# coding: utf-8


import os
import time
import json
import sys
from typing import Any, Iterable, List
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

llm_model = "gpt-3.5-turbo"
PDF_FREELANCER_GUIDELINES_FILE = "./data/프리랜서 가이드라인(출판본).pdf"
CSV_OUTDOOR_CLOTHING_CATALOG_FILE = "./data/OutdoorClothingCatalog_1000.csv"

llm = ChatOpenAI(model_name=llm_model, temperature=0)

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
  


def get_personal_retriever() -> VectorStoreRetriever:
  personal_texts = [
    "내 이름은 홍길동입니다.",
    "내가 제일 좋아하는 색은 보라색입니다.",
    "내 꿈은 최고의 인공지능 활용 어플리케이션 개발자가 되는 것입니다.",
    "내 고향은 제주도입니다.",
    "나는 남성입니다",
    "나는 1972년에 태어났습니다.",
  ]
  personal_retriever = FAISS.from_texts(personal_texts, OpenAIEmbeddings()).as_retriever()
  if not isinstance(personal_retriever, VectorStoreRetriever):
    raise ValueError("personal_retriever is not VectorStoreRetriever")
  return personal_retriever


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

    # 데이터 분석하여 상품 정보 개괄 및 통계 정보 생성 (예: 총 상품수, 성별 의류수, 계절별 의류수 등)
    total_products = len(catalog_df)
    # 여기에 추가적인 통계 정보를 계산하는 코드를 추가

    stats_info = f"안녕하세요. 아웃도어 전문 매장입니다. 현재 저희는 {total_products}개의 제품을 다루고 있으며, 필요한 제품을 자동으로 안내해드리고 있습니다."

    # 임베딩 함수 준비
    embedding = OpenAIEmbeddings()

    # 임베딩 함수와 함께 FAISS 인스턴스 생성
    vectorstore = FAISS.from_texts([stats_info], embedding)

    # 상품 정보 개괄 및 통계 정보를 vectorstore에 추가
    vectorstore.add_texts([stats_info])

    # 채워진 vectorstore를 사용하여 VectorStoreRetriever 인스턴스 생성
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
