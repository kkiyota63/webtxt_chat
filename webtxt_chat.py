import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

#ページの初期設定。タイトルやアイコンを設定。
def init_page():
    st.set_page_config(
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    st.header("Website Summarizer 🤗")
    st.sidebar.title("Options")

#チャットメッセージを初期化。ボタンもこの中で設定。
def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []

#ユーザーが選択した言語モデル（GPT-3.5 または GPT-4）を返す。
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)

#ユーザーからのURL入力を取得。
def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url

#入力されたURLが有効かどうかチェック。
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

#URLからコンテンツをスクレイピング。
def get_content(url):
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # fetch text from main (change the below code to filter page)
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write('something wrong')
        return None

#要約を依頼するためのプロンプトを作成。
def build_prompt(content, n_chars=300):
    return f"""以下はとある。Webページのコンテンツである。内容を{n_chars}程度でわかりやすく要約してください。

========

{content[:1000]}

========

日本語で書いてね！
"""

#GPTモデルによる要約を取得。
def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost

# 1.流れ：
# 2.ページとメッセージを初期化。
# 3.言語モデルを選択。
# 4.URLを入力。
# 5.URLが有効であれば、そのページの内容を取得。
# 6.コンテンツをGPTモデルに要約させる。
# 7. 要約を表示。

def main():
    init_page()

    llm = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Please input valid url')
            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = build_prompt(content)
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = get_answer(llm, st.session_state.messages)
                st.session_state.costs.append(cost)
            else:
                answer = None

    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()