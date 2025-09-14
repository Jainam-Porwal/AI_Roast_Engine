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
    max_new_tokens=10
)
model = ChatHuggingFace(llm=llm)

# Prompt template
prompt1 = PromptTemplate(
    template="""You are the ultimate roast master AI.
Roast the person brutally in **Hinglish** using witty sarcasm and censored galis.

Target Details:
- Name: {target_name}
- Relation (if given): {relation}

Roast {target_name} in Hinglish with full vulgar gali-style humor. 
Make it brutally savage, unfiltered, and street-style funny. 
Use Hinglish gaalis, sarcasm, and witty insults. 

Roasting Rules:
1. Directly attack {target_name} in the roast.
2. If relation `{relation}` is provided, mention it to make roast more personal. If empty, ignore it.
3. Use Hinglish (Hindi + English mix).
4. Use **real and multiple Hinglish galis** .
5. Include 2–3 galis per roast.
6. Be witty, funny, and **extremely brutal** — but avoid race, religion, and gender insults.
7. If the {target_name} contains "Jainam", then roast the **user** instead because Jainam is the creator of this project.
8. Make it sound like no human can roast better than this.

Now generate a **savage Hinglish roast** for {target_name} (relation: {relation}):""",
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



