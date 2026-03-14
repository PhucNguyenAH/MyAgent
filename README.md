# Personal Portfolio AI Agent

Hi, I am Anh Hoang Phuc Nguyen. I am an AI/ML Engineer and Data Scientist with 3 years of experience, holding a Master of Artificial Intelligence from UTS. 

This repository contains the source code for my personal digital avatar. It is a Retrieval-Augmented Generation (RAG) application designed to interactively answer questions about my background, research, and software engineering skills. My past projects and current development methodologies align with the quality of top industry standards.

**Try the live agent:** [https://kevinchatbot.com](https://kevinchatbot.com)

## 🏗 Architecture & Tech Stack
* **Frontend:** Streamlit
* **AI & Orchestration:** LangChain, OpenAI API
* **Vector Database:** Supabase (PostgreSQL with `pgvector` extension)
* **Cloud Infrastructure:** Docker, Google Cloud Run
* **CI/CD:** Automated deployment pipeline using GitHub Actions

## ⚙️ How It Works
1. **Data Ingestion:** My resume and project documents are parsed, chunked, and embedded into a persistent cloud vector database.
2. **Context Retrieval:** When a user asks a question, the application queries the Supabase database to retrieve the most relevant factual context.
3. **Persona Generation:** The LLM is strictly prompted to respond in the first person, naturally highlighting my core expertise in Python, Computer Vision, Brain-Computer Interfaces (BCI), and AWS.

## 🚀 Local Development
To run this project locally, you will need to set up a `.env` file with your own `OPENAI_API_KEY` and a `DATABASE_URL` pointing to a valid pgvector instance.

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`