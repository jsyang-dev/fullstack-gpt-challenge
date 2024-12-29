import streamlit as st
import openai as client
from typing_extensions import override
from openai import AssistantEventHandler
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import yfinance
import json


assistant_id = "asst_f7RNCt80DA1hNgEOHXv7FacN"
# assistant_id = "asst_gLmlNp9EnQaeyt9ale3kpOlc"

st.set_page_config(
    page_title="AssistantGPT",
    page_icon="ℹ️",
)

st.title("AssistantGPT")


class EventHandler(AssistantEventHandler):
    def __init__(self, message_box):
        super().__init__()
        self.message = ""
        self.message_box = message_box

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message)

    @override
    def on_event(self, event):
        if event.event == "thread.run.requires_action":
            with st.status("function 호출중...") as status:
                outputs = get_tool_outputs(event.data.id, event.data.thread_id)
                status.update(label="function 호출 완료", state="complete")

            submit_tool_outputs(
                event.data.id, event.data.thread_id, self.message_box, outputs
            )

    @override
    def on_end(self):
        if self.message:
            save_message(self.message, "assistant")


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def insert_thread(message, role):
    if "thread" not in st.session_state:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": role,
                    "content": message,
                }
            ]
        )
        st.session_state["thread"] = thread
    else:
        thread = st.session_state["thread"]
        client.beta.threads.messages.create(
            thread_id=thread.id, role=role, content=message
        )

    with st.chat_message("assistant") as message_box:
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant_id,
            event_handler=EventHandler(message_box),
        ) as stream:
            stream.until_done()


def send_message(message, role, save=True):
    st.chat_message(role).markdown(message)

    if save:
        save_message(message, role)
        insert_thread(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id, message_box, outputs):
    with client.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(message_box),
    ) as stream:
        stream.until_done()


def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())


def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())


functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}


functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Given the name of a company returns its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API 키를 입력하세요.", type="password")

if openai_api_key:
    send_message("어떤 정보를 알고 싶으신가요?", "assistant", save=False)
    paint_history()
    message = st.chat_input("메시지를 입력하세요.")

    if message:
        send_message(message, "user")

else:
    st.markdown(
        """
        기업에 대해 조사하여 주식을 매수할지 말지 결정할 수 있는 정보를 제공합니다.
                    
        먼저 OpenAI API 키를 입력하세요.
        """
    )

    st.session_state["messages"] = []
