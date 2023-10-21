import os
import time
import json
import asyncio
from typing import List

from dotenv import load_dotenv


from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

import warnings
warnings.filterwarnings('ignore')


def print_start() -> None:
  # 프로그램 시작 시간 기록
  global start_time
  start_time = time.time()
  print("프로그램 실행중...")

def print_end() -> None:
  # 프로그램 종료 시간 기록
  end_time = time.time()
  # 실행 시간 계산
  execution_time = end_time - start_time
  print(f"프로그램 종료: {execution_time} 초")


llm_model = "gpt-3.5-turbo"

load_dotenv()



class MyCustomHandler(BaseCallbackHandler):
  def on_llm_new_token(self, token: str, **kwargs) -> None:
    print(token, end='')


def get_response_schemas() -> List[ResponseSchema]:
  gift_schema = ResponseSchema(name="gift",
                             description="다른 사람을 위한 선물로 구매했는가? \
                              예인 경우 True, \
                              그렇지 아닌경우는 False 값을 설정한다.")
  delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="제품이 도착하는 데 도착하는 데 며칠이 걸렸나? \
                                        이 정보를 찾을 수 없다면 -1 값을 설정한다.")
  price_value_schema = ResponseSchema(name="price_value",
                                    description="값이나 가격에 대한 문장을 추출하여 쉼표로 구분된 Python 목록으로 출력한다.")

  return [gift_schema,
    delivery_days_schema,
    price_value_schema]


def get_parser() -> StructuredOutputParser:
  return StructuredOutputParser.from_response_schemas(get_response_schemas())




def main() -> None:
  handler = MyCustomHandler()
  chat = ChatOpenAI(temperature=0, model=llm_model, streaming=True) # 번역을 항상 같게 하기 위해서 설정

  parser = get_parser()
  format_instructions = parser.get_format_instructions()
  print(format_instructions)

  human_template="""\
    다음의 문장에서 다음과 같은 정보를 추출하라:

    gift: 다른 사람을 위한 선물로 구매했는가? \
          예인 경우 True, 그렇지 아닌경우는 False 값을 설정한다.

    delivery_days: 제품이 도착하는 데 도착하는 데 며칠이 걸렸나? \
                  이 정보를 찾을 수 없다면 -1 값을 설정한다.

    price_value: 값이나 가격에 대한 문장을 추출하여 쉼표로 구분된 Python 목록으로 출력한다.

    문장: {text}

    {format_instructions}
    """
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

  chatchain = LLMChain(llm=chat, prompt=chat_prompt, verbose=True)

  text = """ \
      제 차가 익스플로러 2013년식이라 카플레이 적용되기 전이라서 실시간연동되는 파인드라이브 네비게이션을 쭉 써왔었는데 아무래도 카플레이만큼 편하지는 않더라구요. \
      그리고 최근엔 자꾸 먹통되기도 하고 gps도 잘 못잡고ㅋㅋㅋ
      우연찮게 카플레이 차량 한번 운전했다가 이건 필수다! 싶어서 고민을 했었는데, 빨리 살걸 그랬네요. ㅎㅎ \
      기본내장된 네비 자리에 안드로이드오토교체하려고 알아봐도 저렴한게 50~60이고 거의 백만원 정도라서 포기하고 요걸로 잘 산거 같네요. \
      사진보다 모니터 크기도 꽤 크고 진짜 만족합니다 사은품도 감사합니다.^^ \
      """


  print_start()
  response = chatchain.run(text=text, format_instructions=format_instructions, callbacks=[handler])

  print_end()

  print(response)
  output_dict = parser.parse(response)
  print(output_dict)

  print(output_dict.get('delivery_days'))


if __name__ == '__main__':
  main()