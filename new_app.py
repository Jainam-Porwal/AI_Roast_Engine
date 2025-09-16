import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Hugging Face API setup
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=120,
    temperature=0.9,
)
model = ChatHuggingFace(llm=llm)

# Prompt template for first roast
prompt1 = PromptTemplate(
    template="""You are the ultimate roast master AI.
Roast {target_name} (relation: {relation}) brutally in **Hinglish** with witty sarcasm and censored galis.
Keep it only **3–4 lines**.""",
    input_variables=["target_name", "relation"]
)

parser = StrOutputParser()
chain = prompt1 | model | parser

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "target_locked" not in st.session_state:
    st.session_state.target_locked = False
if "target_name" not in st.session_state:
    st.session_state.target_name = ""
if "relation" not in st.session_state:
    st.session_state.relation = ""

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Hinglish Roast AI", page_icon="🤖")

# Custom CSS with transparent bubbles
st.markdown("""
    <style>
    .chat-box {
        max-height: 450px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #444;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: rgba(0, 102, 204, 0.3); /* blue tint */
        border: 1px solid #1a73e8;
        color: #fff;
        padding: 10px 14px;
        border-radius: 15px;
        margin: 6px;
        text-align: right;
        max-width: 75%;
        float: right;
        clear: both;
        font-size: 15px;
        font-weight: 500;
    }
    .ai-bubble {
        background-color: rgba(20, 20, 20, 0.7); /* dark transparent */
        border: 1px solid #666;
        color: #fff; /* white text */
        padding: 10px 14px;
        border-radius: 15px;
        margin: 6px;
        text-align: left;
        max-width: 75%;
        float: left;
        clear: both;
        font-size: 15px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)
# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align: center;'>🤖 Hinglish Roast AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>😡 Roast Your Target • Then Keep Chatting 🔥</p>", unsafe_allow_html=True)

# ---------------- STEP 1: CHOOSE TARGET ----------------
if not st.session_state.target_locked:
    st.subheader("🎯 Choose Your Target First")

    cols = st.columns([2, 1])
    target_name = cols[0].text_input("Target Name:", placeholder="Kis par gussa aa raha hai? 😡")
    relation = cols[1].text_input("Relation:", placeholder="Dost, BF, GF, Teacher, Ex, etc.")
    
    if st.button("🔥 Roast Now"):
        if target_name.strip() == "":
            st.warning("⚠️ Naam to likh bhai!")
        else:
            with st.spinner("Cooking roast... 🍳"):
                roast = chain.invoke({"target_name": target_name, "relation": relation or "unknown"})
                # Save target + chat state
                st.session_state.target_name = target_name
                st.session_state.relation = relation or "unknown"
                st.session_state.chat_history.append(AIMessage(content=roast))
                st.session_state.target_locked = True
                st.rerun()

# ---------------- STEP 2: CHAT MODE ----------------
else:
    st.subheader(f"💬 Roasting {st.session_state.target_name} ({st.session_state.relation})")

    # Chat messages
    chat_container = st.container()
    with chat_container:
        st.markdown("<div class='chat-box' id='chat-box'>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.markdown(f"<div class='user-bubble'>👤 {msg.content}</div>", unsafe_allow_html=True)
            elif isinstance(msg, AIMessage):
                st.markdown(f"<div class='ai-bubble'>🤖 {msg.content}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Auto-scroll to bottom
        st.markdown("""
            <script>
                var chatBox = document.getElementById('chat-box');
                chatBox.scrollTop = chatBox.scrollHeight;
            </script>
        """, unsafe_allow_html=True)

    # Input bar (always at bottom)
    with st.form(key="chat_form", clear_on_submit=True):
        user_msg = st.text_input("Type your roast or reply:", placeholder="Aur kya bole isko? 😂")
        submitted = st.form_submit_button("👉 Send")
        if submitted and user_msg.strip():
            st.session_state.chat_history.append(HumanMessage(content=user_msg))
            with st.spinner("Roasting harder... 🔥"):
                result = model.invoke(st.session_state.chat_history)
                st.session_state.chat_history.append(AIMessage(content=result.content))
                st.rerun()

    # Restart button
    if st.button("🔄 Roast Someone Else"):
        st.session_state.chat_history = []
        st.session_state.target_locked = False
        st.session_state.target_name = ""
        st.session_state.relation = ""
        st.rerun()


