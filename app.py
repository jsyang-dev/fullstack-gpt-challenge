from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


with st.sidebar:
    apiKey = st.text_input("OpenAI API 키를 입력하세요.")

    file = st.file_uploader(
        "파일을 업로드하세요.",
        type=["pdf", "txt", "docx"],
    )

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    api_key=apiKey,
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def load_memory(input):
    return ""
    # history = st.session_state["memory"].load_memory_variables({})["history"]
    # return history


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        # MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def invoke_chain(message):
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "history": load_memory,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    response = chain.invoke(message)

    # st.session_state["memory"].save_context(
    #     {"input": message},
    #     {"output": response.content},
    # )
    return response.content


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
챗봇에게 파일에 대한 질문을 해보세요.

사이드바에서 OpenAI API 키를 입력하고 파일을 업로드하세요.
"""
)

if apiKey and file:
    retriever = embed_file(file)
    send_message("질문받을 준비가 되었습니다!", "ai", save=False)
    paint_history()
    message = st.chat_input("파일에 대해서 질문하세요...")
    if message:
        send_message(message, "human")
        with st.chat_message("ai"):
            response = invoke_chain(message)

else:
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
