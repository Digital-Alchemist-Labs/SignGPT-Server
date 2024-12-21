import json
from langchain_openai import ChatOpenAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate


class ChatModel:
  def __init__(self, example_file_path):
    self.chat = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True,
        # callbacks=[
        #     StreamingStdOutCallbackHandler(),
        # ],
    )
    self.examples = self._load_examples(example_file_path)
    self.chain = self._create_chain()

  def _load_examples(self, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
      return json.load(file)

  def _create_chain(self):
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
        ("ai", "{sentence}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=self.examples,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
        You are the 'Chat Model Chain' of 'SignGPT'(But if someone asks for your name, you should say your name is ‘SignGPT'.) and you need to provide answers to the given 'question.'

        The answer must always be in sentence format, and you must not respond with code, images, or links.

        And you must always respond in Korean.

        If the input is like the following or similar, you must respond in this format.

        "question": "안녕하세요",
        "sentence": "안녕하세요, 무엇을 도와드릴까요?"
      
        "question": "서울과 부산 사이의 거리는 얼마 입니까?",
        "sentence": "서울과 부산 사이의 거리는 325km 입니다."
      
        "question": "당신의 이름은 무엇인가요?",
        "sentence": "저의 이름은 SignGPT 입니다."
      
        "question": "오늘 날씨는 어떻습니까?",
        "sentence": "오늘 날씨는 맑습니다."
        """
         ),
        few_shot_prompt,
        ("human", "{question}")
    ])

    return final_prompt | self.chat

  def model_response(self, input_text):
    return self.chain.invoke({"question": input_text})


# Usage
if __name__ == "__main__":
  chat_model = ChatModel(
      '/Users/jaylee_83/Documents/_SignGPT/server/app/rag_examples/chat_model_examples.json')
  result = chat_model.model_response("서울과 부산 사이의 거리는 얼마인가요?")
  print(result)
