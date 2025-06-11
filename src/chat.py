import streamlit as st

from dotenv import load_dotenv
from llm import get_ai_response, debug_prompt_preview

load_dotenv()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("🤖소득세 챗봇")
st.caption("소득세와 관련된 모든 것을 답해드립니다!")

# Initialization
if 'messageList' not in st.session_state:
    st.session_state['messageList'] = []

print(f"===before=== {st.session_state['messageList']}")

for message in st.session_state['messageList']:
    with st.chat_message(message["role"]):
        print(f"message: {message}")
        st.write(message["content"])

if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요! "):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state['messageList'].append({"role": "user", "content":user_question})

    with st.spinner("답변을 처리하는 중 입니다"):
        ai_response = get_ai_response(user_question)
        debug_prompt_preview(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state['messageList'].append({"role": "ai", "content":ai_message})
print(f"===after=== {st.session_state['messageList']}")

