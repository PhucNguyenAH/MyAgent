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
import dotenv

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gpt-5.4", model_provider="openai")

# Connect to the existing database (does NOT add new documents)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=os.environ.get("DATABASE_URL"),
    use_jsonb=True,
)

# rag_prompt = hub.pull("rlm/rag-prompt")
# Define your custom persona and instructions
system_instructions = (
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

# Create the prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    ("human", "{question}"),
])

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def response_generator(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)

if __name__ == "__main__":
    st.title("Anh Hoang Phuc Nguyen Chatbot")

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Fixed variable shadowing by using user_query instead of prompt
    if user_query := st.chat_input("What do you want to ask about the applicant?"):
        with st.chat_message("user"):
            st.markdown(user_query)
            
        st.session_state.messages.append({"role": "user", "content": user_query})
        response = graph.invoke({"question": user_query})

        with st.chat_message("assistant"):
            st.write_stream(response_generator(response['answer']))
    
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})