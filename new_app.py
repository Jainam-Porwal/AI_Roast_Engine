import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
    st.subheader(f"💬 Roasting Session with {st.session_state.target_name} ({st.session_state.relation})")

    # Show past chat
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(
                f"<div style='background-color:#f1f1f1; padding:10px; border-radius:10px; margin:5px; text-align:right;'>"
                f"👤 <b>You:</b> {msg.content}</div>",
                unsafe_allow_html=True
            )
        elif isinstance(msg, AIMessage):
            st.markdown(
                f"<div style='background-color:#e6f0ff; padding:10px; border-radius:10px; margin:5px; text-align:left;'>"
                f"🤖 <b>Roast AI:</b> {msg.content}</div>",
                unsafe_allow_html=True
            )

    # Chat input at bottom
    with st.form(key="chat_form", clear_on_submit=True):
        user_msg = st.text_input("Type your roast or reply:", placeholder="Aur kya bole isko? 😂")
        submitted = st.form_submit_button("👉 Send")
        if submitted and user_msg.strip():
            st.session_state.chat_history.append(HumanMessage(content=user_msg))
            with st.spinner("Roasting harder... 🔥"):
                result = model.invoke(st.session_state.chat_history)
                st.session_state.chat_history.append(AIMessage(content=result.content))
                st.rerun()

    # Option to restart with new person
    if st.button("🔄 Roast Someone Else"):
        st.session_state.chat_history = []
        st.session_state.target_locked = False
        st.session_state.target_name = ""
        st.session_state.relation = ""
        st.rerun()
