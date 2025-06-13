import streamlit as st
from llm import get_aimessage

st.set_page_config(page_title='크리에이터를 위한 유튜브 가이드라인 챗봇', page_icon='🎥')

st.title('유튜브 가이드라인 챗봇 🎥')
st.markdown("### for 'YouTube 크리에이터'")


query = "다른 나라 언어를 자막으로 넣는 방법을 알려주세요."

if inputchat := st.chat_input(placeholder="유튜브 크리에이터 가이드라인에 대해 질문해주세요"):
    with st.chat_message("user"):
        st.write(f"{inputchat}")

    aimessage = get_aimessage(inputchat)
    with st.chat_message("ai"):        
        st.write(aimessage)

