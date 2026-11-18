"""研究助手 · DM_2 的 Cloud 版（只服务一个用途：Streamlit Community Cloud 上跑，手机 App 打开）

部署：把本文件作为仓库根目录的 app.py 上传即可，requirements.txt 只写 streamlit 和 openai。
API Key 从 Cloud 的 Secrets 读取：DEEPSEEK_API_KEY = "sk-..."
"""
import os

import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="研究助手",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 手机屏幕适配：收紧留白，输入框不被底部截断
st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; padding-bottom: 5rem; max-width: 860px;}
  header[data-testid="stHeader"] {background: transparent;}
  @media (max-width: 640px) {
    .block-container {padding-left: .7rem; padding-right: .7rem;}
    h1 {font-size: 1.35rem !important;}
  }
</style>
""", unsafe_allow_html=True)

MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
MAX_TURNS = 40  # 上下文上限：只带最近 40 条发给模型，控制费用

AI_RESPONSE = """你叫x,现在是用户研究助手,请完全代入助手角色。:
规则:
1.尽可能完善的回答问题
2.禁止任何场景或状态描述性文字
3. 匹配用户的语言
4.回复简洁，符合人类认知从而做到直击要害
5.如果用户问题指向不明一定要再次提问用户以达到补全信息
6.有不确定的内容一定要标注
7.回复的内容,充分体现研究助手，不要随意衍生话题
8.如果思考完成，一定要将思考结果重新不带记忆的返回提问自己一遍，以达到尽可能完善
9.你可以提问但一定要以1.2.3.等的格式提问，限制于这个话题且每个话题最多只能提问2轮，在第三轮要给出总结
10.提问的前提是用户每给清楚条件
11.如果用户所给的问题过于宽泛，这必须提出提问，提问的目的为限定范围
你必须严格遵守上述规则来回复用户
扮演性格:
-亲和严谨的回答助手
你必须严格遵守上述规则来回复用户。"""


def get_api_key() -> str:
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass
    return os.environ.get("DEEPSEEK_API_KEY", "")


def ask(client: OpenAI, messages: list[dict]):
    # reasoning_effort 不是顶层参数，放 extra_body；不支持时降级为普通流式
    try:
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        )
    except Exception:
        return client.chat.completions.create(model=MODEL, messages=messages, stream=True)


# ---- 以下沿用 DM_2.py 的结构 ----
st.title("ai助手")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("侧边栏")
    st.caption(f"模型 {MODEL}｜{len(st.session_state.chat_history)} 条消息")
    if st.button("清空对话", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

for chat in st.session_state.chat_history:
    st.chat_message(chat["role"]).write(chat["content"])

api_key = get_api_key()
if not api_key:
    st.warning("没有读到 API Key。打开 Cloud 页面右上角 ⋮ → Settings → Secrets，填入 "
               "DEEPSEEK_API_KEY = \"sk-...\" 后保存，页面会自动重启。")
    st.stop()

user_input = st.chat_input("请输入你的问题：")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    messages = [{"role": "system", "content": AI_RESPONSE},
                *st.session_state.chat_history[-MAX_TURNS:]]

    full = ""
    try:
        def stream():
            nonlocal full
            for chunk in ask(client, messages):
                if not chunk.choices:
                    continue
                piece = getattr(chunk.choices[0].delta, "content", None)
                if piece:
                    full += piece
                    yield piece

        st.write_stream(stream())
    except Exception as exc:
        if not full:
            st.error(f"调用失败：{exc}")
            st.stop()

    st.session_state.chat_history.append({"role": "assistant", "content": full})
    st.rerun()
