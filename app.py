import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)
model = genai.GenerativeModel("gemini-2.5-flash")
embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)
st.set_page_config(
    page_title="AI StudyBuddy",
    page_icon="📚",
    layout="wide"
)
# CUSTOM CSS
st.markdown("""
<style>

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Buttons */
.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    font-weight: 600;
}

/* Tabs container */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    justify-content: center;
}

/* Individual tabs */
.stTabs [data-baseweb="tab"] {
    height: 60px;
    min-width: 170px;
    white-space: nowrap;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: 600;
}

/* Success box */
.stAlert {
    border-radius: 10px;
}

/* Metric cards */
[data-testid="metric-container"] {
    border-radius: 12px;
    padding: 15px;
    background-color: #111827;
    border: 1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# HEADER
col1, col2 = st.columns([4,1])

with col1:
    st.markdown("""
    <h1 style='text-align: center;'>
    📚 AI StudyBuddy
    </h1>

    <h4 style='text-align: center; color: gray;'>
    Upload study material and learn smarter!
    </h4>
    """, unsafe_allow_html=True)

with col2:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
        width=130
    )

# SIDEBAR
with st.sidebar:
    st.header("📖 Features")

    st.info("📌 AI Summary")

    st.info("🧠 Quiz Generator")

    st.info("🃏 Smart Flashcards")

    st.info("❓ Doubt Solver")

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "📂 Upload PDF",
    type="pdf"
)

text = ""

# AFTER PDF UPLOAD
if uploaded_file:
    with st.spinner("Uploading PDF..."):
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
                
        # Limit token usage
        text = text[:5000]
    st.success("✅ PDF Uploaded Successfully!")
    
    # METRICS
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Pages", len(pdf_reader.pages))
    with col2:
        st.metric("📝 Characters", len(text))
    with col3:
        st.metric("🤖 AI Status", "Ready")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Summary",
        "🧠 Quiz",
        "🃏 Flashcards",
        "❓ Doubt Solver"
    ])

    # SUMMARY TAB
    with tab1:
        st.subheader("📌 Generate Summary")
        if st.button("Generate Summary"):
            prompt = f"""
            Summarize the following study material.

            Rules:
            - Use simple language
            - Use bullet points
            - Keep important concepts

            Content:
            {text}
            """

            try:
                with st.spinner("Generating Summary..."):
                    response = model.generate_content(prompt)
                st.success("✅ Summary Generated!")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
    # QUIZ TAB
    with tab2:
        st.subheader("🧠 Generate Quiz")
        if st.button("Generate Quiz"):
            prompt = f"""
            You are an AI teacher.

            Create 5 multiple-choice questions.

            STRICT RULES:
            - Questions should test understanding
            - Each question must have 4 options
            - Each option MUST be on separate lines
            - Add blank lines between options
            - Mention correct answer clearly
            - Keep formatting clean

            FORMAT:

            Question 1:
            What is AI?

            A) Artificial Intelligence

            B) Artificial Interface

            C) Automatic Internet

            D) Advanced Input

            Correct Answer: A)

            Study Material:
            {text}
            """
            try:
                with st.spinner("Generating Quiz..."):
                    response = model.generate_content(prompt)
                st.success("✅ Quiz Generated!")
                st.text(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
    # FLASHCARDS TAB
    with tab3:
        st.subheader("🃏 Generate Flashcards")
        if st.button("Generate Flashcards"):
            prompt = f"""
            Create flashcards from this study material.

            FORMAT:

            Q:
            A:
            
            Content:
            {text}
            """

            try:
                with st.spinner("Generating Flashcards..."):
                    response = model.generate_content(prompt)
                st.success("✅ Flashcards Generated!")
                st.text(response.text)
            except Exception as e:
                st.error(f"Error: {e}")

    # DOUBT SOLVER TAB
    with tab4:
        st.subheader("❓ Ask Questions from PDF")
        # Better chunking
        chunks = [
            chunk for chunk in text.split(".")
            if chunk.strip()
        ]

        # Embeddings
        chunk_embeddings = embedding_model.encode(chunks)
        # FAISS index
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(chunk_embeddings))
        # User question
        user_question = st.text_input(
            "Ask your doubt"
        )

        if user_question:
            try:
                with st.spinner("Finding Answer..."):
                    # Question embedding
                    question_embedding = embedding_model.encode(
                        [user_question]
                    )

                    # Search similar chunks
                    D, I = index.search(
                        np.array(question_embedding),
                        k=2
                    )

                    relevant_text = ""
                    for idx in I[0]:
                        relevant_text += chunks[idx]
                    # Final prompt
                    prompt = f"""
                    Answer ONLY from the provided context.

                    Context:
                    {relevant_text}

                    Question:
                    {user_question}

                    If answer is unavailable,
                    say:
                    "Answer not found in uploaded material."
                    """
                    response = model.generate_content(prompt)
                st.success("✅ Answer Generated!")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")


# FOOTER
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<hr>

<center>
Built with ❤️ by <b>Nagaraj S</b> 🚀
</center>
""", unsafe_allow_html=True)