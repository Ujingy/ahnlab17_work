#!/usr/bin/env python
# coding: utf-8

# # LangChain: Agents
#
# ## Outline:
#
# * Using built in LangChain tools: DuckDuckGo search and Wikipedia
# * Defining your own tools


#!pip install -U wikipedia

from dotenv import load_dotenv

llm_model = "gpt-3.5-turbo"

load_dotenv()



from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0, model=llm_model)


tools = load_tools(["llm-math","wikipedia"], llm=llm)


agent= initialize_agent(
  tools,
  llm,
  agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  handle_parsing_errors=True,
  verbose = True)


result = agent.run("페이스북 창업자는 누구인지? 그의 현재(2023년) 나이는? 그의 현재 나이를 제곱하면?")

print(result)

print(agent.tools)


# ## Python Agent

agent = create_python_agent(
  llm,
  tool=PythonREPLTool(),
  verbose=True
)

customer_list = [["Harrison", "Chase"],
         ["Lang", "Chain"],
         ["Dolly", "Too"],
         ["Elle", "Elem"],
         ["Geoff","Fusion"],
         ["Trance","Former"],
         ["Jen","Ayai"]
        ]


result = agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")

print(result)


import langchain
langchain.debug=True
result = agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")
langchain.debug=False

print(result)


# ## Define your own tool

#!pip install DateTime

from langchain.agents import tool
from datetime import date


@tool
def time(text: str) -> str:
  """Returns todays date, use this for any \
  questions related to knowing todays date. \
  The input should always be an empty string, \
  and this function will always return todays \
  date - any date mathmatics should occur \
  outside this function."""
  return str(date.today())


agent= initialize_agent(
  tools + [time],
  llm,
  agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  handle_parsing_errors=True,
  verbose = True)




try:
  result = agent("whats the date today?")
  print(result)
except:
  print("exception on external access")


