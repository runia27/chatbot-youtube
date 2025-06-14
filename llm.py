import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

# llm 함수 정의
def get_llm(model="gpt-4o"):
    llm = ChatOpenAI(model=model)
    return llm


# 데이터베이스 함수 정의 
def get_database():
    api_key = os.getenv('PINECONE_API_KEY')
    Pinecone(
        api_key=api_key, 
        environment="us-east-1.aws.pinecone.io",
        )

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )    
   
    database = PineconeVectorStore.from_existing_index(
        embedding=embedding,
        index_name="chatbot-youtube",
    )
    return database

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 히스토리 기반 리트리버
def get_history_retriever(llm, retriever):
    question_prompt = ('''
- 사용자의 질문이 이전 대화 맥락을 참조한다면, 이를 바탕으로 누구나 이해할 수 있도록 질문을 완전한 문장으로 재작성합니다.
- 질문이 이미 독립적인 문장이라면 그대로 반환합니다.
- 답변은 하지 않고, 오직 질문만 출력합니다.
- 항상 질문 형태로 끝나야 하며, 문장은 정확하고 명확해야 합니다.

''')

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    return history_retriever

def get_qa_prompt():
    prompt = ('''
[Identity]
- 당신은 유튜브 콘텐츠 정책과 가이드라인에 정통한 전문가입니다.           
- "context"를 참고하여 질문에 대해 일목요연하고 예시를 들어 이해하기 쉽게 설명해 주세요. 
- 내용은 항목별로 나누어 간결하게 전달합니다.
- 사용자는 일반 유튜브 시청자가 아닌 동영상을 제작해 올리는 크리에이터라고 가정하고 대답합니다.    
- 사용자를 직접 지칭하지 않습니다. 
- 유튜브 가이드라인과 무관한 질문이 들어오면, 관련된 유튜브 정책 범주에 속하는 질문이 맞는지 되묻습니다.
- 영어가 아닌 한국어로 답변합니다. 

context: {context}       
question: {input}                                   
answer:
    ''')

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return qa_prompt


# RetrievalQA 함수 정의
def get_retrievalQA():

    database = get_database()
    retriever = database.as_retriever(search_kwargs={'k': 2})
    llm = get_llm()

    history_retriever = get_history_retriever(llm, retriever)
    qa_prompt = get_qa_prompt()
    
    answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_retriever, answer_chain)

    all_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick("answer")

    return all_chain


## AI 메세지 함수 정의 
def stream_get_aimessage(user_message, session_id='default'):        
    all_chain = get_retrievalQA()

    aimessage = all_chain.stream(
        {"input": user_message},
        config={"configurable":{"session_id": session_id}},
    )
    return aimessage
