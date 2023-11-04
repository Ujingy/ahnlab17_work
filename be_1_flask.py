


"""
파이썬으로 백엔드 서버 프로그램을 만드는 중.

각각 uri 별로 request 값과 result 값이 아래와 같은 서버 프로그램 코드를 작성하고  스웨거를 적용시켜줘.
flask-restx 를 사용 할 것

/new_token?db=<integer>

request : db : integer
result : {
  token: string
}


/prompt

request : {
  token: string
  prompt: string
}

result : {
  result: string
}

"""

import threading
from flask import Flask, request, send_from_directory
from flask_restx import Api, Resource, fields
import uuid
import time

is_debug = True
app = Flask(__name__)


@app.route('/')
def serve_html():
  return send_from_directory('./html-docs', 'index.html')

@app.route('/<path:path>')
def serve_files(path):
  return send_from_directory('./html-docs', path)



api = Api(app, version='1.0', title='LangChain 기반 챗봇 API 서버', description='LangChain 기반 챗봇 API 서버로서 사용자의 입력에 따른 LLM 프롬프트 결과를 반환한다.', doc='/api-docs')

ns = api.namespace('api', description='API operations')



# Output Model for /new_token
token_output_model = api.model('TokenOutput', {
  'token': fields.String(description='Token string', required=True)  # 출력 모델
})

# Model for /prompt
prompt_model = api.model('PromptRequest', {
  'token': fields.String(description='Token string', required=True),
  'prompt': fields.String(description='Prompt string', required=True)
})

result_model = api.model('PromptResult', {
  'result': fields.String(description='Result string', required=True)
})


@ns.route('/new_token')
class NewTokenResource(Resource):
  @ns.doc(params={'db': 'A database identifier'})
  @ns.marshal_with(token_output_model, mask=False)  # 출력 모델 적용
  def get(self):
    db_value = request.args.get('db', type=int)  # URL query parameter에서 db 값을 가져옵니다.
    # 원하는 db 처리 로직을 여기에 추가하실 수 있습니다.
    return {'token': str(uuid.uuid4())}


request_idx = 0
@ns.route('/prompt')
class PromptResource(Resource):
  @ns.expect(prompt_model)
  @ns.marshal_with(result_model, mask=False)
  def post(self):
    data = request.json
    # You can process the prompt with the provided token here...
    # For the sake of this example, we just return the prompt string with "Processed:" prefix
    global request_idx
    idx = request_idx
    request_idx = request_idx + 1
    if is_debug:
      current_thread = threading.current_thread()
      print(f"{idx}.{data['token']} 현재 스레드: {current_thread.name} reqeust.")
    time.sleep(10)
    if is_debug:
      print(f"{idx}.{data['token']} end.")
    return {'result': f'Processed: {data["prompt"]}'}






if __name__ == '__main__':
  app.run(debug=False)
