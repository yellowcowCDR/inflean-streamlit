from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

# 프롬프트 구성
from langchain_core.messages import HumanMessage, AIMessage
# ChatPromptTemplate (qa_prompt) 가져오기

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_rag_chain():
    llm = get_llm()

    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요."
        "아래에 제공된 소득세법 전문을 바탕으로, 사용자의 질문에 해당하는 내용이 어느 조문에 포함되어 있는지를 판단하고 답변해주세요."
        "반드시 “소득세법 제XX조에 따르면,“으로 답변을 시작해주세요."
        "사용자의 질문이 해당 법령에 포함되지 않거나 근거가 불명확한 경우에는 “모르겠습니다” 또는 “해당 내용은 소득세법에서 명시적으로 다루고 있지 않습니다”라고 답변해주세요."
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    history_aware_retriever = get_history_retriever()

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain.pick("answer")

def get_retreiver(k=4):
    # 3-1. 임베딩
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # 3-2. 벡터 DB에 저장
    index_name = 'tax-index2'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retreiver = database.as_retriever(search_kwargs={"k": k})
    return retreiver

def get_history_retriever():
    llm = get_llm()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = get_retreiver(4)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

def get_llm(model="gpt-4o"):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = "사람이 나타나는 표현 -> 거주자"
    prompt = ChatPromptTemplate.from_template("""
           사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
           만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
           그런 경우에는 질문만 리턴해주세요.  
           사전: {dictionary}

           질문: {question}

        """
                                              )
    llm = get_llm()
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain



def get_ai_response(user_message):
    # 5. 유사도 검색으로 가져온 문서를 LLM 질문과 같이 전달
    dictionary = "사람이 나타나는 표현 -> 거주자"
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    tax_chain = {"input": dictionary_chain} | rag_chain
    ai_response = tax_chain.stream(
        {
            "question": user_message,
            "dictionary": dictionary
        },
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )

    return ai_response


def debug_prompt_preview(user_message: str):
    dictionary = "사람이 나타나는 표현 -> 거주자"
    dictionary_chain = get_dictionary_chain()

    # 사용자 질문 수정
    revised_question = dictionary_chain.invoke({
        "question": user_message,
        "dictionary": dictionary
    })

    print("\n📌 사용자 질문(사전 반영 후):", revised_question)

    # 프롬프트 생성 (QA Prompt만 대상)
    retriever = get_retreiver()
    llm = get_llm()

    # 문서 검색
    documents = retriever.invoke(revised_question)


    chat_history = [
        HumanMessage(content="퇴직금은 소득세가 부과되나요?"),
        AIMessage(content="소득세법 제12조에 따르면, 퇴직소득은 분리과세 대상으로 일반적인 소득세 부과 대상이 아닙니다."),
    ]



    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{answer}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요."
        "아래에 제공된 소득세법 전문을 바탕으로, 사용자의 질문에 해당하는 내용이 어느 조문에 포함되어 있는지를 판단하고 답변해주세요."
        "반드시 “소득세법 제XX조에 따르면,“으로 답변을 시작해주세요."
        "사용자의 질문이 해당 법령에 포함되지 않거나 근거가 불명확한 경우에는 “모르겠습니다” 또는 “해당 내용은 소득세법에서 명시적으로 다루고 있지 않습니다”라고 답변해주세요."
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    formatted_messages = qa_prompt.format_messages(
        input=revised_question,
        context="\n".join(doc.page_content for doc in documents),
        chat_history=chat_history
    )

    print("\n\n✅ 최종 생성된 프롬프트 메시지:")
    for msg in formatted_messages:
        print(f"[{msg.type.upper()}] {msg.content}")