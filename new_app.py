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

# Prompt template (for initial roast)
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

# ----------------- UI START -----------------
st.set_page_config(page_title="Hinglish Roast AI", page_icon="🔥")

st.markdown("<h1 style='text-align: center;'>🤖🔥 Hinglish Roast AI 🔥🤖</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>😈 Brutal, Savage & Funny Hinglish Roasts 😈</p>", unsafe_allow_html=True)

# Roast section
st.header("🎯 Roast Generator")
target_name = st.text_input("Target Name 🧑:", placeholder="👀 Kis par gussa aa raha hai?")
relation = st.text_input("Relation 🤔:", placeholder="⚡ Uska rishta kya hai tumse?")
if st.button("💥 Roast Now!"):
    if target_name.strip() == "":
        st.warning("⚠️ Pehle naam to likh bhai!")
    else:
        with st.spinner("😈 Tandoor mein roast ho raha hai..."):
            relation_text = relation if relation.strip() else "unknown"
            roast = chain.invoke({
                "target_name": target_name,
                "relation": relation_text
            })
            st.session_state.chat_history.append(AIMessage(content=roast))
            st.success(f"🔥 Savage Roast for **{target_name}** 🔥\n\n{roast}")

# Chat section
st.header("💬 Chat with Roast Master AI")
user_input = st.text_input("Type your reply here 📝:", key="chat_input", placeholder="Batao kya bolu ab? 🤔")
if st.button("👉 Send"):
    if user_input.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        if user_input.lower() == "exit":
            st.warning("🚪 Chat ended! Refresh to start again.")
        else:
            result = model.invoke(st.session_state.chat_history)
            st.session_state.chat_history.append(AIMessage(content=result.content))

# Display chat history with bubbles
st.subheader("📜 Chat History")
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"<div style='background-color:#d1f7c4;padding:10px;border-radius:10px;margin:5px;text-align:right;'>👤 <b>You:</b> {msg.content}</div>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f"<div style='background-color:#ffe4e1;padding:10px;border-radius:10px;margin:5px;text-align:left;'>🤖 <b>Roast AI:</b> {msg.content}</div>", unsafe_allow_html=True)
