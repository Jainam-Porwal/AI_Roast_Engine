import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Hugging Face API setup
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") , # ✅ Fixed
    max_new_tokens=100,       # ✅ must be here
    temperature=0.9, 
    max_length=256, 

)
model = ChatHuggingFace(llm=llm)

# Prompt template
prompt1 = PromptTemplate(
    template="""You are the ultimate roast master AI.
Roast the person brutally in **Hinglish** using witty sarcasm and censored galis.
Make it brutally savage, unfiltered, and street-style funny. 
 in just 3 to 4 line only .
Now generate a **savage Hinglish roast** for {target_name} (relation: {relation}) in 3 to 4 line only:""",
    input_variables=["target_name", "relation"]
)

chain = prompt1 | model

# Streamlit UI
st.title("🔥 Hinglish Roast Generator 🔥")
st.write("Get a brutally savage roast in Hinglish with witty sarcasm!")

# Input fields
target_name = st.text_input("Enter target name:")
relation = st.text_input("Relation with the Target name (optional):")

if st.button("Roast Now!"):
    if target_name.strip() == "":
        st.warning("⚠️ Please enter a target name first!")
    else:
        with st.spinner("Cooking up some savage roast..."):
            roast = chain.invoke({
                "target_name": target_name,
                "relation": relation
            })
            st.subheader("Here’s the Roast:")

            st.write(roast.content)










