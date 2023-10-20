import time
import os
import copy
from typing import Iterable, List
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.schema.vectorstore import VectorStore

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings

class ExecutionTimeChecker:
  def __init__(self, with_start: bool = False):
    """
    ExecutionTimeChecker 클래스의 생성자입니다.

    :param with_start: 시간 측정을 시작할지 여부를 결정하는 부울 값
    """
    self.start_time = 0
    self.end_time = 0
    if with_start:
      self.start_time = time.time()

  def start(self) -> 'ExecutionTimeChecker':
    """
    시간 측정을 시작하고, self를 반환합니다.

    :return: self (ExecutionTimeChecker 객체)
    """
    self.start_time = time.time()
    self.end_time = 0
    return self

  def stop(self) -> float:
    """
    시간 측정을 종료하고 경과 시간을 반환합니다.

    :return: 경과 시간 (초 단위)
    """
    self.end_time = time.time()
    return self.get_elapsed_time()

  def get_elapsed_time(self) -> float:
    """
    시간 측정 결과인 경과 시간을 반환합니다.

    :return: 경과 시간 (초 단위)
    """
    if self.start_time:
      current_time = time.time()
      if self.end_time:
        return self.end_time - self.start_time
      else:
        return current_time - self.start_time
    else:
      return None



class ConfigManager:
  default_config = None
  def __init__(self, config = {}):
    """
    ConfigManager 클래스의 생성자입니다. 이 클래스는 설정 항목을 관리합니다.

    :param config: 초기 설정 항목을 담긴 딕셔너리 또는 기본값 ( 설정되지 않은 경우 )
    """
    self.config = copy.deepcopy(config)

  def set(self, key, value):
    """
    설정 항목을 설정합니다.

    :param key: 설정 항목의 키
    :param value: 설정 항목의 값
    """
    self.config[key] = value

  def get(self, key, default=None):
    """
    설정 항목의 값을 가져옵니다.

    :param key: 설정 항목의 키
    :param default: 설정 항목이 없을 경우 반환할 기본값 (선택 사항)
    :return: 설정 항목의 값 또는 기본값 (설정되지 않은 경우)
    """
    return self.config.get(key, default)

  def get_all(self):
    """
    모든 설정 항목을 가져옵니다.

    :return: 모든 설정 항목을 담은 딕셔너리
    """
    return self.config

  @classmethod
  def get_env(cls, key, default=None):
    """
    환경 변수에서 설정 항목의 값을 가져옵니다.

    :param key: 설정 항목의 키
    :param default: 설정 항목이 없을 경우 반환할 기본값 (선택 사항)
    :return: 환경 변수에서 가져온 설정 항목의 값 또는 기본값 (설정되지 않은 경우)
    """
    return os.environ.get(key, default)

  @classmethod
  def get_default_config(cls, config = {}):
    """
    클래스의 기본 설정(ConfigManager 객체)을 가져옵니다. 기본 설정은 클래스 레벨에서 공유되며,
    필요한 경우 주어진 구성(config)으로 초기화됩니다.

    :param config: 초기 설정을 나타내는 딕셔너리 (선택 사항)
    :return: ConfigManager 객체
    """
    if not cls.default_config:
      cls.default_config = ConfigManager(config)

    return cls.default_config


def get_filename_without_extension(file_path):
  """
  주어진 파일 경로에서 확장자를 제외한 파일명을 추출합니다.

  :param file_path: 파일 경로
  :return: 확장자를 제외한 파일명
  """
  # 파일 경로를 파싱하여 파일명과 확장자로 분리
  file_name, file_extension = os.path.splitext(os.path.basename(file_path))

  return file_name


def split_docs(docs: Iterable[Document]) -> List[Document]:
  """
  문서를 분할하는 함수입니다.

  :param docs: Document 객체의 Iterable
  :return: 분할된 Document 객체의 List
  """
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
  )

  splits = text_splitter.split_documents(docs)
  # print(f"len(splits)=>{len(splits)}")
  return splits


def load_pdf(file: str, with_split: bool = False) -> List[Document]:
  """
  PDF 파일을 로드하는 함수입니다.

  :param file: PDF 파일의 경로
  :param with_split: 문서 분할 여부를 결정하는 부울 값
  :return: Document 객체의 List
  """
  loader = PyPDFLoader(file)
  pages = loader.load()
  if not with_split:
    return pages

  return split_docs(pages)



def load_pdfs( files: Iterable[str], with_split: bool = False) -> List[Document]:
  """
  PDF 파일들을 로드하는 함수입니다.

  :param files: PDF 파일의 목록
  :param with_split: 문서 분할 여부를 결정하는 부울 값
  :return: Document 객체의 List
  """

  docs = []
  for file in files:
    docs.extend(load_pdf(file, with_split))

  return docs




def get_pdf_vectordb_path(file: str) -> str:
  """
  PDF 벡터 데이터베이스(.vectordb) 파일의 경로를 반환합니다.

  :param file: PDF 파일의 경로 또는 파일명
  :return: PDF 벡터 데이터베이스 파일의 경로
  """
  # 파일명에서 확장자를 제외한 부분 추출
  file_name = get_filename_without_extension(file)

  # 설정된 VECTORDBPATH 환경 변수 또는 기본 경로를 사용하여 PDF 벡터 데이터베이스 파일 경로 생성
  path = os.path.join(ConfigManager.get_env("VECTORDBPATH", "./vectordb"), f"pdf/{file_name}")

  return path

def save_vectordb(embedding: Embeddings, db_path: str, documents: List[Document]) -> VectorStore:
  """
  벡터 데이터베이스를 저장하고 반환합니다.

  :param embedding: Embeddings 객체
  :param db_path: 벡터 데이터베이스 파일 경로
  :param documents: 저장할 문서 리스트
  :return: 저장된 VectorStore 객체
  """
  # 문서와 임베딩을 사용하여 벡터 데이터베이스 생성
  vectordb = FAISS.from_documents(
    documents=documents,
    embedding=embedding
  )

  # 로컬 파일로 벡터 데이터베이스 저장
  vectordb.save_local(db_path)

  return vectordb

def load_vectordb(embedding: Embeddings, db_path: str) -> VectorStore:
  """
  벡터 데이터베이스를 로드하고 반환합니다.

  :param embedding: Embeddings 객체
  :param db_path: 벡터 데이터베이스 파일 경로
  :return: 로드된 VectorStore 객체
  """
  # 로컬 파일로부터 벡터 데이터베이스 로드
  vectordb = FAISS.load_local(embeddings=embedding, folder_path=db_path)

  return vectordb

def load_pdf_vectordb(file: str) -> VectorStore:
  """
  PDF 벡터 데이터베이스를 로드하거나 생성하고 반환합니다.

  :param file: PDF 파일의 경로 또는 파일명
  :return: 로드된 또는 생성된 VectorStore 객체
  """
  # PDF 벡터 데이터베이스 파일 경로 가져오기
  db_path = get_pdf_vectordb_path(file)

  vectordb: VectorStore = None
  documents: List[Document] = None

  # 벡터 데이터베이스 파일이 존재하지 않으면 PDF를 로드하고 벡터 데이터베이스 생성
  if not os.path.exists(db_path):
    documents = load_pdf(file, True)
    vectordb = save_vectordb(OpenAIEmbeddings(), db_path, documents)
  else:
    # 벡터 데이터베이스 파일이 이미 존재하면 로드
    vectordb = load_vectordb(OpenAIEmbeddings(), db_path)

  return vectordb
