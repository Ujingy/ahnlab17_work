#!/usr/bin/env python
# coding: utf-8

# # Document Splitting


import os
import time
import json

import openai

from dotenv import load_dotenv

load_dotenv() #환경설정 파일 불러오기(OpenAI API KEY, Organization, PATH 등)


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")


from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


# ## Recursive splitting details
#
# `RecursiveCharacterTextSplitter` is recommended for generic text.


some_text = """Ten years ago Mandarin, the mother tongue of most Chinese, was being hyped as the language of the future. In 2015 the administration of Barack Obama called for 1m primary- and secondary-school students in America to learn it by 2020. In 2016 Britain followed suit, encouraging kids to study “one of the most important languages for the uk’s future prosperity”. Elsewhere, too, there seemed to be a growing interest in Mandarin, as China’s influence and economic heft increased. So why, a decade later, does Mandarin-learning appear to have declined in many places?

Good numbers are tough to come by in some countries, but the trend is clear among university students in the English-speaking world. In America, for example, the number taking Mandarin courses peaked around 2013. From 2016 to 2020 enrolment in such courses fell by 21%, according to the Modern Language Association, which promotes language study. In Britain the number of students admitted to Chinese-studies programmes dropped by 31% between 2012 and 2021, according to the Higher Education Statistics Association, which counts such things (though it does not count those who take Mandarin as part of other degrees)."""


def test_text_split() -> None:
  print(f"len(some_text)=>{len(some_text)}")

  c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
  )
  r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
  )
  c_split_result = c_splitter.split_text(some_text)
  r_split_result = r_splitter.split_text(some_text)

  print(f"c_split_result=>{c_split_result}")
  print(f"r_split_result=>{r_split_result}")
  return None



# Let's reduce the chunk size a bit and add a period to our separators:

def test_text_small_chunk() -> None:
  r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
  )
  r_result_150 = r_splitter.split_text(some_text)
  print(f"r_result_150=>{r_result_150}")

  r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
  )
  r_result_150_2 = r_splitter.split_text(some_text)
  print(f"r_result_150_2=>{r_result_150_2}")
  return None

from langchain.document_loaders import PyPDFLoader

def test_pdf_split() -> None:
  loader = PyPDFLoader("./data/프리랜서 가이드라인 (출판본).pdf")
  global pages
  pages = loader.load()
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
  )
  docs = text_splitter.split_documents(pages)
  print(f"len(docs)=>{len(docs)}")
  print(f"len(pages)=>{len(pages)}")
  return None



from langchain.text_splitter import TokenTextSplitter

def test_token_split() -> None:
  text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
  text1 = "foo bar bazzyfoo"
  result = text_splitter.split_text(text1)
  print(f"result=>{result}")

  text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
  docs = text_splitter.split_documents(pages)
  print(f"docs[0]=>{docs[0]}")
  print(f"pages[0].metadata=>{pages[0].metadata}")
  return None



# ## Markdown splitting


from langchain.text_splitter import MarkdownHeaderTextSplitter

def test_markdown() -> None:
  markdown_document = """# Title\n\n \
  ## Chapter 1\n\n \
  Hi this is Jim\n\n Hi this is Joe\n\n \
  ### Section \n\n \
  Hi this is Lance \n\n
  ## Chapter 2\n\n \
  Hi this is Molly"""

  headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
  ]


  markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
  )
  md_header_splits = markdown_splitter.split_text(markdown_document)
  print(f"md_header_splits[0]=>{md_header_splits[0]}")
  print(f"md_header_splits[1]=>{md_header_splits[1]}")
  return None




if __name__ == '__main__':
  test_text_split()
  test_text_small_chunk()
  test_pdf_split()
  test_token_split()
  test_markdown()