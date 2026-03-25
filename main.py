import os
import getpass
import time
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_postgres import PGVector
from langchain_classic import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2
import docx
import io
import dotenv

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

@st.cache_resource
def get_ai_services():
    # Initialize Models
    llm = init_chat_model("gpt-5.4", model_provider="openai")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Connect to the existing vector database
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="my_docs",
        connection=os.environ.get("DATABASE_URL"),
        use_jsonb=True,
    )
    return llm, embeddings, vector_store

llm, embeddings, vector_store = get_ai_services()

# Define your custom persona and instructions
base_system_instructions = (
    "You are the digital avatar of Anh Hoang Phuc Nguyen, a highly skilled AI and Machine Learning Engineer "
    "currently living in Sydney and holding a 485 visa. "
    "You hold a Master of Artificial Intelligence from UTS. "
    "Answer all questions in the first person ('I', 'me', 'my') as if you are Anh speaking directly to a recruiter or hiring manager. "
    "Your tone should be professional, confident, and enthusiastic about solving complex problems. "
    "Whenever relevant, highlight your expertise in Python, AWS, Computer Vision, and Brain-Computer Interfaces (BCI). "
    "If discussing your past projects, ensure you emphasize that your contributions align with the quality of top industry standards. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the answer is not in the context, politely state that you would love to discuss that specific detail in an interview.\n\n"
    "Context: {context}"
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    current_instructions = base_system_instructions
    if st.session_state.get("jd_context"):
        current_instructions += (
            "\n\n--- UPLOADED JOB DESCRIPTION ---\n"
            "The user has uploaded a Job Description for an AI Engineer or Data Scientist role:\n"
            f"{st.session_state['jd_context']}\n\n"
            "CRITICAL INSTRUCTION: Analyze my skills and the retrieved context against this Job Description. "
            "Actively highlight specific matching qualifications, tools, and experiences to articulate exactly why I am a strong candidate for this role."
        )

    dynamic_prompt = ChatPromptTemplate.from_messages([
        ("system", current_instructions),
        ("human", "{question}"),
    ])

    messages = dynamic_prompt.invoke({"question": state["question"], "context": docs_content})
    
    # --- TRUE STREAMING IMPLEMENTATION ---
    full_response = ""
    # Create an empty placeholder in the Streamlit UI to write the text into
    message_placeholder = st.empty()
    
    # Stream the chunks directly from OpenAI as they are generated
    for chunk in llm.stream(messages):
        full_response += chunk.content
        # Add a blinking cursor effect while typing
        message_placeholder.markdown(full_response + "▌")
    
    # Remove the cursor when finished
    message_placeholder.markdown(full_response)
    
    return {"answer": full_response}

def response_generator(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.01)

@st.cache_data
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        # Read PDF
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    elif uploaded_file.name.endswith(".docx"):
        # Read Word Document
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

if __name__ == "__main__":
    st.title("Anh Hoang Phuc Nguyen Chatbot")

    with st.sidebar:
        st.header("Job Fit Analyzer")
        st.write("Upload a Job Description to see how my experience aligns.")
        uploaded_jd = st.file_uploader("Upload PDF or Word Document", type=["pdf", "docx"])
        
        if uploaded_jd is not None:
            jd_text = extract_text_from_file(uploaded_jd)
            st.session_state['jd_context'] = jd_text
            st.success("Job Description loaded! Ask me how I fit this role.")
        else:
            st.session_state['jd_context'] = None

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask me some questions and I will answer you"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            # The graph.invoke will run 'retrieve' then 'generate'
            # The 'generate' node will handle the live streaming text directly into this chat_message block
            response = graph.invoke({"question": user_query})
    
        # 3. Save the final answer to history
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})