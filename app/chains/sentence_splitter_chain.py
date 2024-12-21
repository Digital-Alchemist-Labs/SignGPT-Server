import json
from langchain_openai import ChatOpenAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate


class SentenceSplitter:
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
        ("human", "{sentence}"),
        ("ai", "{result}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=self.examples,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are the ‘sentence splitter’ and your role is to split the input sentence into words separated by ‘,’ according to sign language grammar. To split a Korean sentence according to sign language grammar. You must respond in the format of the few-shot prompt.
            
            If the input is like the following or similar, you must respond in this format.

            "sentence": "안녕하세요, 무엇을 도와드릴까요?",
            "result": "안녕하세요, 돕다, 무엇"

            "sentence": "서울과 부산 사이의 거리는 약 325km입니다.",
            "result": "서울, 부산, 거리, 325, km, 입니다"

            "sentence": "저의 이름은 SignGPT 입니다.",
            "result": "나, 이름, SignGPT, 입니다"

            "sentence": "오늘 날씨는 맑습니다.",
            "result": "오늘, 날씨, 맑다"
            """
        ),
        few_shot_prompt,
        ("human", "{sentence}")
    ])

    return final_prompt | self.chat

  def split_sentence(self, input_text):
    return self.chain.invoke({"sentence": input_text})


# Usage
if __name__ == "__main__":
  chat_model = SentenceSplitter(
      '/Users/jaylee_83/Documents/_SignGPT/server/app/rag_examples/sentence_splitter_examples.json')
  result = chat_model.split_sentence("서울과 부산 사이의 거리는 325km입니다.")
  print(result)
  result = chat_model.chain.invoke({"sentence": "제 이름은 SignGPT입니다."})
  print(result)
