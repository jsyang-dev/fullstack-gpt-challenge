import json

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
from pathlib import Path


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API 키를 입력하세요.", type="password")

    level = st.selectbox("레벨을 선택하세요.", ("Easy", "Medium", "Hard"))


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def get_level(_):
    return level


function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

if openai_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        api_key=openai_api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )

    questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant that is role playing as a teacher.

                Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.

                Questions are created at selected level. It is possible to have 3 levels: Easy, Medium and Hard.

                Each question should have 4 answers, three of them must be incorrect and one should be correct.

                Your turn!

                Level: {level}

                Context: {context}
                """,
            )
        ]
    )

    questions_chain = (
        {"context": format_docs, "level": get_level} | questions_prompt | llm
    )


@st.cache_data(show_spinner="파일 로딩중...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    Path("./.cache/quiz_files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="위키피디아 검색중...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="퀴즈 생성중...")
def run_quiz_chain(_docs, key):
    response = questions_chain.invoke(_docs)
    return json.loads(response.additional_kwargs["function_call"]["arguments"])


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "원하는 방법을 선택하세요.",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "파일 업로드",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
        여러분의 지식을 테스트하고 공부하는 데 도움이 되도록, 업로드한 파일이나 위키피디아 문서를 바탕으로 퀴즈를 만들어 드립니다.
                    
        1. OpenAI API 키를 입력하세요.
        2. 레벨을 선택하세요.
        3. 파일을 업로드하거나 위키피디아 문서를 검색하세요.
        """
    )
else:
    if choice == "File":
        response = run_quiz_chain(docs, file.name)
    else:
        response = run_quiz_chain(docs, topic)

    with st.sidebar:
        grade_button = st.button("채점하기", key="grade_button")

    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"]):
            st.write(question["question"])
            value = st.radio(
                "선택하세요.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"question_{idx}",
            )

            if grade_button:
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("정답")
                elif value is not None:
                    st.error("오답")
        st.form_submit_button("제출하기")

    with st.sidebar:
        if grade_button:
            correct = 0
            wrong = 0

            for question in response["questions"]:
                if {"answer": value, "correct": True} in question["answers"]:
                    correct += 1
                elif value is not None:
                    wrong += 1

            st.write(f"정답: {correct}")
            st.write(f"오답: {wrong}")

            question_length = len(response["questions"])
            if correct + wrong == question_length:
                if correct == question_length:
                    st.success("만점입니다!!")
                    st.balloons()
                elif value is not None:
                    st.error(f"{wrong}개 문제를 틀렸습니다.")
            else:
                st.warning("모든 문제를 풀고 제출해주세요.")
