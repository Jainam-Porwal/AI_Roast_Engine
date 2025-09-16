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
    max_new_tokens=100,
    temperature=0.9,
)
model = ChatHuggingFace(llm=llm)

# Prompt template (for roasting)
prompt1 = PromptTemplate(
    template="""You are the ultimate roast master AI.
Roast the person brutally in **Hinglish** using witty sarcasm and censored galis.
Keep it only **3–4 lines**.

Now generate a savage Hinglish roast for {target_name} (relation: {relation}):""",
    input_variables=["target_name", "relation"]
)

parser = StrOutputParser()
chain = prompt1 | model | parser

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a savage Hinglish roaster AI 🤖🔥")
    ]

# Page config
st.set_page_config(page_title="Hinglish Roast AI", page_icon="🤖")

# ----------- UI -----------
st.markdown("<h1 style='text-align: center;'>🤖 Hinglish Roast AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>😡 Release Your Anger • Roast Them All 🔥</p>", unsafe_allow_html=True)

# Display chat history first
st.subheader("📜 Chat History")
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(
                f"<div style='background-color:#f1f1f1; padding:10px; border-radius:10px; margin:5px; text-align:right;'>"
                f"<b>You:</b> {msg.content}</div>",
                unsafe_allow_html=True
            )
        elif isinstance(msg, AIMessage):
            st.markdown(
                f"<div style='background-color:#e6f0ff; padding:10px; border-radius:10px; margin:5px; text-align:left;'>"
                f"<b>Roast AI 🤖:</b> {msg.content}</div>",
                unsafe_allow_html=True
            )

# ---------------- Chat Input (BOTTOM like ChatGPT) ----------------
st.markdown("---")
st.markdown("### 💬 Roast Someone Now")

with st.form(key="roast_form", clear_on_submit=True):
    cols = st.columns([2, 1])  # target name wider
    target_name = cols[0].text_input("Target Name:", placeholder="Kis par gussa aa raha hai? 😡")
    relation = cols[1].text_input("Relation:", placeholder="Dost, BF, GF, Teacher, Ex, etc.")
    submitted = st.form_submit_button("🔥 Roast Now")

    if submitted:
        if target_name.strip() == "":
            st.warning("⚠️ Pehle naam to likh bhai!")
        else:
            with st.spinner("Cooking roast... 🍳"):
                relation_text = relation if relation.strip() else "unknown"
                roast = chain.invoke({
                    "target_name": target_name,
                    "relation": relation_text
                })
                # Add both Human request + AI roast to chat
                st.session_state.chat_history.append(HumanMessage(content=f"Roast {target_name} ({relation_text})"))
                st.session_state.chat_history.append(AIMessage(content=roast))
