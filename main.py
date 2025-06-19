import streamlit as st
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv
import os
import asyncio
import re  
import unicodedata

def sanitize_filename(filename):
    """Special characters do handle and safe create filename"""
    # Step 1: Unicode normalization
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    # Step 2: Special characters remove
    filename = re.sub(r'[^\w\s-]', '', filename).strip()
    # Step 3: Spaces/underscores handle
    filename = re.sub(r'[-\s]+', '_', filename)
    # Step 4: Windows-specific illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    return filename[:50]  # Max length limit

# Page settings
st.set_page_config(page_title="Essay Agent", page_icon=":globe_with_meridians:")

st.markdown("""
<style>
    /* Button styling */
    .stButton>button {
        background-color: #1f77b4;
        color: white !important;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1a66a0 !important;
        color: white !important;
    }
    /* Text area styling */
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
</style> 
""", unsafe_allow_html=True)

# Load environment variables and handle API key
load_dotenv()
try:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets["OPENROUTER_API_KEY"]
except Exception:
    openrouter_api_key = None

if not openrouter_api_key:
    st.error("""
        API key not found! Either:
        1. For local development: Set OPENROUTER_API_KEY in .env file
        2. For deployment: Add OPENROUTER_API_KEY to Streamlit secrets
    """)
    st.stop() 

async def generate_essay(topic: str):
    """Generate essay based on user topic"""
    external_client = AsyncOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    model = OpenAIChatCompletionsModel(
        model="deepseek/deepseek-r1-0528:free",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    agent = Agent(
        name="Writer Agent",
        instructions="You are a professional writer. Generate well-structured essays in simple English with proper introduction, body paragraphs and conclusion."
    )

    result = await Runner.run(
        agent,
        input=f"Write a comprehensive essay about: {topic}",
        run_config=config
    )
    return result.final_output

def main():
    st.title("📝 Smart Essay Agent")
    st.markdown("Enter your topic below and get a well-written essay instantly!")
    
    # User input section
    with st.form("essay_form"):
        essay_topic = st.text_area(
            "What would you like your essay about?",
            placeholder="e.g., The importance of education, Allama Iqbal's philosophy, Climate change effects...",
            height=150
        )
        submit_button = st.form_submit_button("Generate Essay")
    
    if submit_button and essay_topic:
        with st.spinner(f"Generating essay about '{essay_topic}'..."):
            try:
                essay = asyncio.run(generate_essay(essay_topic))
                
                st.success("Your essay is ready!")
                st.markdown("---")
                st.subheader(f"Essay: {essay_topic}")
                st.markdown(essay)
                
                # Add download button
                safe_filename = sanitize_filename(essay_topic)
                file_name = f"essay_{safe_filename[:50]}.txt"
                
                st.download_button(
                    label="Download Essay",
                    data=essay,
                    file_name=file_name,
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    elif submit_button and not essay_topic:
        st.warning("Please enter a topic for your essay")

if __name__ == "__main__":
    main() 