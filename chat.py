import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone


load_dotenv()

api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

database = PineconeVectorStore(
    embedding=embedding,
    index_name='chatbot-youtube'
)

llm = ChatOpenAI()

prompt = hub.pull("rlm/rag-prompt")

## RetrievalQA 구현
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

chain_lcel = (
    {
        "context" : database.as_retriever() | format_docs,
        "question" : RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

aimessage = chain_lcel.invoke("다른 나라 언어를 자막으로 넣는 방법을 알려주세요.")

print(aimessage)
