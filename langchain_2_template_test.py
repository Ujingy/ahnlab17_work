import time

from dotenv import load_dotenv


from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)

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


def main() -> None:
  chat = ChatOpenAI(temperature=0, model=llm_model) # 번역을 항상 같게 하기 위해서 설정

  template="You are a helpful assisstant that tranlates {input_language} to {output_language}." #프롬프트 언어 변경
  system_message_prompt = SystemMessagePromptTemplate.from_template(template)

  human_template="{text}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

  chatchain = LLMChain(llm=chat, prompt=chat_prompt, verbose=True)
  text = """
독점: 미국, 이란에서 압수한 무기를 우크라이나로 이송할 예정

미국 당국자들에 따르면, 미국은 이란에서 압수한 수천 개의 무기와 탄약을 우크라이나로 이송할 예정입니다. 이러한 조치는 우크라이나
 군대가 미국과 동맹국으로부터 더 많은 자금과 장비를 기다리는 동안, 우크라이나 군대가 직면한 심각한 부족 상황을 완화하는 데 도움
이 될 수 있습니다.
"""

  print_start()
  response = chatchain.run(input_language="Korean", output_language="English", text=text)

  print_end()

  print(response)


if __name__ == '__main__':
  main()