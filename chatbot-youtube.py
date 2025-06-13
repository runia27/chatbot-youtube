import streamlit as st
from llm import get_aimessage

st.set_page_config(page_title='크리에이터를 위한 유튜브 가이드라인 챗봇', page_icon='🎥')

st.title('유튜브 가이드라인 챗봇 🎥')
st.markdown("### for 'YouTube 크리에이터'")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for i in st.session_state.message_list:
    with st.chat_message(i['role']):
        st.write(i['content'])


if inputchat := st.chat_input(placeholder="유튜브 크리에이터 가이드라인에 대해 질문해주세요"):
    with st.chat_message("user"):    
        st.write(f"{inputchat}")
    st.session_state.message_list.append({'role': 'user', 'content': inputchat})

    with st.spinner("답변을 생성하는 중입니다."):
        aimessage = get_aimessage(inputchat)
        
        with st.chat_message("ai"):
            st.write(aimessage)
        st.session_state.message_list.append({'role': 'ai', 'content': aimessage})

