import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# -------------------------------------------------
# 🔑 Robust API KEY LOADING (FIXED)
# -------------------------------------------------

load_dotenv()

groq_api_key = (
    os.getenv("GROQ_API_KEY")  # local .env or system env
    or st.secrets.get("GROQ_API_KEY", None)  # Streamlit Cloud secrets
)

if not groq_api_key:
    st.error(
        "❌ GROQ_API_KEY not found.\n\n"
        "• Local: add to .env file\n"
        "• Streamlit Cloud: Manage App → Secrets → GROQ_API_KEY"
    )
    st.stop()

# Initialize LLM safely
langchain_llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile"
)

# -------------------------------------------------
# PROMPTS
# -------------------------------------------------

plan_prompt_template = """
You are a fitness and diet planner. Using the following inputs, create two detailed plans:
1. A **diet plan** table listing day-to-day food intake for {number_of_weeks} weeks.
2. A **workout plan** table listing day-to-day exercises for {number_of_weeks} weeks.

Inputs:
- **Workout type**: {workout_type}
- **Diet type**: {diet_type}
- **Current body weight**: {current_weight} kg
- **Target weight**: {target_weight} kg
- **Specific dietary restrictions**: {dietary_restrictions}
- **Health conditions**: {health_conditions}
- **Age**: {age}
- **Gender**: {gender}
- **Other instructions**: {comments}

Return the plans in a neat, structured format with tables and include any relevant key notes.
"""

plan_prompt = PromptTemplate(
    input_variables=[
        "workout_type",
        "diet_type",
        "current_weight",
        "target_weight",
        "dietary_restrictions",
        "health_conditions",
        "age",
        "gender",
        "number_of_weeks",
        "comments",
    ],
    template=plan_prompt_template,
)

chat_prompt_template = """
You are a fitness and diet expert. Answer the following user question based on the given plan:

Plan: {plan}

Question: {question}

Provide a clear and helpful response.
"""

chat_prompt = PromptTemplate(
    input_variables=["plan", "question"],
    template=chat_prompt_template,
)

# -------------------------------------------------
# CHAINS
# -------------------------------------------------

plan_chain = plan_prompt | langchain_llm | StrOutputParser()
chat_chain = chat_prompt | langchain_llm | StrOutputParser()

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

st.set_page_config(page_title="🧘‍♀️ Fitness and Diet Planner", layout="wide")
st.title("🧘‍♀️ Fitness and Diet Planner")

col1, col2 = st.columns(2)

# ---------------- Column 1 ----------------
with col1:
    st.header("Enter your details:")

    workout_type = st.text_input("Workout Type (e.g., Weight Loss, Muscle Gain)")
    diet_type = st.text_input("Diet Type (e.g., Indian, Mediterranean)")
    current_weight = st.number_input("Current Body Weight (kg)", 30.0, 200.0, 75.0)
    target_weight = st.number_input("Target Weight (kg)", 30.0, 200.0, 68.0)
    dietary_restrictions = st.text_input("Dietary Restrictions")
    health_conditions = st.text_input("Any Health Conditions?")
    age = st.number_input("Age", 10, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    number_of_weeks = st.slider("Number of Weeks", 1, 12, 4)
    comments = st.text_area("Additional Comments")

    if st.button("Generate Plans"):
        st.session_state["messages"] = []

        with st.spinner("Generating personalized fitness and diet plans..."):
            try:
                response = plan_chain.invoke({
                    "workout_type": workout_type,
                    "diet_type": diet_type,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "dietary_restrictions": dietary_restrictions,
                    "health_conditions": health_conditions,
                    "age": age,
                    "gender": gender,
                    "number_of_weeks": number_of_weeks,
                    "comments": comments,
                })

                st.session_state.plan = response
                st.success("Plans generated successfully!")

            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- Column 2 ----------------
with col2:
    if "plan" in st.session_state:
        st.header("Your Plans:")
        st.markdown(
            f'<div class="scrollable-response">{st.session_state.plan}</div>',
            unsafe_allow_html=True
        )

# ---------------- Chat ----------------
if "plan" in st.session_state:
    st.markdown("---")
    st.subheader("Converse with your plan")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question about your plan"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            answer = chat_chain.invoke({
                "plan": st.session_state.plan,
                "question": prompt
            })
        except Exception as e:
            answer = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

# Footer
st.markdown("---")
st.caption("Saptarshi Ghosh")

