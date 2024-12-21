from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from chains.sentence_finisher_chain import SentenceFinisher
from chains.chat_model_chain import ChatModel
from chains.sentence_splitter_chain import SentenceSplitter

load_dotenv('/Users/itsjay.83/Documents/code/git_clone/SignGPT-Server/.env')

openai_api_key = os.getenv('OPENAI_API_KEY')

# chat = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
#     api_key=openai_api_key
# )

# _SentenceFinisher = SentenceFinisher(
#     example_file_path='/Users/jaylee_83/Documents/_DigitalAlchemistLabs/SignGPT/server/app/rag_examples/sentence_finisher_example.json')

# _ChatModel = ChatModel(
#     example_file_path='/Users/jaylee_83/Documents/_DigitalAlchemistLabs/SignGPT/server/app/rag_examples/chat_model_examples.json')

# _SentenceSplitter = SentenceSplitter(
#     example_file_path='/Users/jaylee_83/Documents/_DigitalAlchemistLabs/SignGPT/server/app/rag_examples/sentence_splitter_examples.json')

_SentenceFinisher = SentenceFinisher(
    example_file_path='/Users/itsjay.83/Documents/code/git_clone/SignGPT-Server/app/rag_examples/sentence_finisher_example.json')

_ChatModel = ChatModel(
    example_file_path='/Users/itsjay.83/Documents/code/git_clone/SignGPT-Server/app/rag_examples/chat_model_examples.json')

_SentenceSplitter = SentenceSplitter(
    example_file_path='/Users/itsjay.83/Documents/code/git_clone/SignGPT-Server/app/rag_examples/sentence_splitter_examples.json')

# 1. sentence finisher

sentence_finisher_chain = _SentenceFinisher.chain

chat_model_chain = _ChatModel.chain

sentence_splitter_chain = _SentenceSplitter.chain

sign_gpt_chain = sentence_finisher_chain | chat_model_chain | sentence_splitter_chain

app = FastAPI(
    title="Sign GPT LLM Serve",
    version="0.1.0",
    description="test",
)


@app.get("/")
async def redirect_root_to_docs():
  return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, sign_gpt_chain, path="/sgc")
add_routes(app, sentence_finisher_chain, path="/sfc")
add_routes(app, chat_model_chain, path="/cmc")
add_routes(app, sentence_splitter_chain, path="/ssc")

if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)
