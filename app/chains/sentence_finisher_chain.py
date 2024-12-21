import json
from langchain_openai import ChatOpenAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate


class SentenceFinisher:
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
        ("human", "{words}"),
        ("ai", "{question}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=self.examples,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are a 'sentence finisher' that completes sentences based on recognized Korean Sign Language (KSL) words. When KSL words are input, construct the sentence according to KSL grammar rules.
            The KSL words will be input separated by ','.
            If a '?' is at the end of a word, the sentence becomes a question. If '끝' appears at the end of a word, the sentence is in the past tense.
            Always generate a Korean sentence.
         
         If the input is like the following or similar, you must respond in this format.
 
          "words": "안녕하세요",
          "question": "안녕하세요"
        
          "words": "서울, 부산, 거리, ?",
          "question": "서울과 부산 사이의 거리는 얼마 입니까?"
        
          "words": "너, 이름, ?",
          "question": "당신의 이름은 무엇인가요?"
        
          "words": "오늘, 날씨, ?",
          "question": "오늘 날씨는 어떻습니까?"

         """),
        few_shot_prompt,
        ("human", "{words}"),
    ])

    return final_prompt | self.chat

  def finish_sentence(self, input_text):
    return self.chain.invoke({"words": input_text})


# Usage
if __name__ == "__main__":
  finisher = SentenceFinisher(
      '/Users/jaylee_83/Documents/_SignGPT/server/app/rag_examples/sentence_finisher_example.json')
  result = finisher.finish_sentence("서울, 부산, 거리, ?")
  print(result)
  result = finisher.chain.invoke({"question": "서울, 부산, 거리, ?"})
  print(result)
