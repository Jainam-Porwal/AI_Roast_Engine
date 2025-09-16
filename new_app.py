import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
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
    max_length=256,
)
model = ChatHuggingFace(llm=llm)

# Prompt template
prompt1 = PromptTemplate(
    template="""You are the ultimate roast master AI.
Roast the person brutally in **Hinglish** using witty sarcasm and censored galis.
Make it brutally savage, unfiltered, and street-style funny. 
In just 3 to 4 lines only.
Now generate a savage Hinglish roast for {target_name} (relation: {relation}) in 3 to 4 lines only:""",
    input_variables=["target_name", "relation"]
)

parser = StrOutputParser()
chain = prompt1 | model | parser

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "roasting" not in st.session_state:
    st.session_state.roasting = False
if "target" not in st.session_state:
    st.session_state.target = ""
if "relation" not in st.session_state:
    st.session_state.relation = ""

# Title
st.title("🔥 Roast AI - Hinglish Savage Mode 🔥")

# If roasting mode not started
if not st.session_state.roasting:
    target_name = st.text_input("👉 Kis par gussa aa raha hai? (Target Name)", placeholder="e.g. Jainam")
    relation = st.text_input("👉 Relation (dost, BF, GF, teacher, ex, etc.)", placeholder="e.g. Dost")

    if st.button("Roast Now! 🔥"):
        if target_name.strip() == "":
            st.warning("⚠️ Please enter a target name first!")
        else:
            with st.spinner("Cooking up some savage roast..."):
                roast = chain.invoke({
                    "target_name": target_name,
                    "relation": relation
                })
                st.session_state.chat_history.append(("AI", roast))
                st.session_state.roasting = True
                st.session_state.target = target_name
                st.session_state.relation = relation

# If roasting mode active
else:
    st.subheader(f"💬 Roasting Session with {st.session_state.target} ({st.session_state.relation})")

    # Show chat history (simple style, no CSS)
    for sender, msg in st.session_state.chat_history:
        if sender == "AI":
            st.markdown(f"**🤖 Roast AI:** {msg}")
        else:
            st.markdown(f"**🧑 You:** {msg}")

    # Chat input
    user_input = st.text_input("Type your roast or reply:", placeholder="Aur kya bole isko? 😆")
    if st.button("👉 Send"):
        if user_input.strip() != "":
            st.session_state.chat_history.append(("You", user_input))
            with st.spinner("AI roasting back..."):
                response = model.invoke([HumanMessage(content=user_input)])
                st.session_state.chat_history.append(("AI", response.content))
            st.experimental_rerun()

    # Restart option
    if st.button("🔄 Roast Someone Else"):
        st.session_state.chat_history = []
        st.session_state.roasting = False
        st.session_state.target = ""
        st.session_state.relation = ""
        st.experimental_rerun()
