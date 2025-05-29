import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import webbrowser

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible for the provided context. 
    If the answer is not in the provided context, just say "Answer is not available in the context".
    Context:\n {context}?\n
    Question:\n {question}?\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[-1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = "".join([item["text"] for item in transcript_text])
        return transcript
    except Exception as e:
        raise Exception("Error extracting transcript: " + str(e))

def generate_gemini_summary(transcript_text, prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        if not transcript_text or not prompt:
            return "Error: Missing transcript or prompt"
            
        response = model.generate_content(prompt + transcript_text)
        
        if hasattr(response, 'text'):
            return response.text
        return "Sorry, couldn't generate summary."
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        st.info("Please verify your API key and internet connection")
        return None

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        if not docs:
            st.warning("No relevant information found in the uploaded PDFs for the given question.")
            return
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {e}")

def check_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 Google API Key not found!")
        st.markdown("""
        ### How to fix this:
        1. Create a `.env` file in your project directory
        2. Add your API key: `GOOGLE_API_KEY=your_key_here`
        3. Make sure the API key is from [Google AI Studio](https://makersuite.google.com/app/apikey)
        """)
        st.stop()
    return api_key

def display_feature_cards():
    """Display feature cards for current application"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            margin-bottom: 20px;
        ">
            <div style="font-size: 2.5em; margin-bottom: 15px;">🎥</div>
            <h3 style="margin: 0 0 10px 0; font-size: 1.3em;">YouTube Summarizer</h3>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">Extract and summarize key insights from any YouTube video using AI-powered transcript analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            margin-bottom: 20px;
        ">
            <div style="font-size: 2.5em; margin-bottom: 15px;">📚</div>
            <h3 style="margin: 0 0 10px 0; font-size: 1.3em;">PDF Chat Assistant</h3>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">Upload multiple PDFs and ask intelligent questions. Get instant answers from your documents</p>
        </div>
        """, unsafe_allow_html=True)

def create_modern_selector():
    """Create a modern tab-like selector"""
    st.markdown("""
    <style>
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        font-weight: 600;
        font-size: 16px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def display_navigation_section():
    """Display navigation section with links to other applications"""
    st.markdown("---")
    
    st.markdown("## 🚀 Explore More AI Tools")
    st.markdown("Discover additional AI-powered applications to boost your productivity")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Enhanced card design for SQL Generator
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 30px;
            border-radius: 20px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
            margin: 20px 0;
        ">
            <div style="font-size: 3em; margin-bottom: 20px;">🔍</div>
            <h3 style="margin: 0 0 15px 0; font-size: 1.5em;">SQL Query Generator</h3>
            <p style="margin: 0 0 25px 0; opacity: 0.95; font-size: 1em;">
                Generate SQL queries from natural language using AI
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple button without complex styling
        sql_url = "https://sqlquerygenerator-7xbeqcgnivqtywjghc4sd3.streamlit.app/"
        
        if st.button("🚀 Launch SQL Generator", key="sql_generator_btn", use_container_width=True):
            st.markdown(f'<meta http-equiv="refresh" content="0; url={sql_url}">', unsafe_allow_html=True)
            st.success("Redirecting to SQL Generator...")
        
        # Enhanced features list
        with st.expander("✨ SQL Generator Features", expanded=False):
            st.markdown("""
            - 📊 **Natural Language to SQL**: Convert plain English questions to SQL queries
            - 📁 **Excel File Support**: Upload and query Excel data directly  
            - 🤖 **AI-Powered**: Uses Google Gemini AI for intelligent query generation
            - 💾 **Instant Results**: Execute queries and view results immediately
            - 🔒 **Secure**: Safe handling of your data with proper sanitization
            """)

def main():
    api_key = check_api_key()
    genai.configure(api_key=api_key)
    st.set_page_config(
        page_title="AI Document & Video Suite", 
        layout="wide", 
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced main title with gradient background
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        color: white;
    ">
        <h1 style="font-size: 3em; margin-bottom: 15px; font-weight: 700;">
            🧠 AI Document & Video Suite
        </h1>
        <p style="font-size: 1.3em; opacity: 0.9; margin: 0;">
            Powered by Google Gemini AI - Your Intelligent Assistant for Documents and Videos
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display feature cards upfront
    st.markdown("## 🎯 Current Application Features")
    st.markdown("Choose from our powerful AI-driven tools below")
    
    display_feature_cards()

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing Google API Key. Please set it in the .env file.")
        return

    # Create the selector with YouTube as default (index 0)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        option = st.selectbox(
            "🎯 Select AI Tool:",
            ["Summarize YouTube Video", "Chat with PDFs"],
            index=0,
            help="Select the AI feature you want to use"
        )

    # Enhanced styling for main content area
    st.markdown("---")

    if option == "Summarize YouTube Video":
        st.markdown("### 🎥 YouTube Video Summarizer")
        st.markdown("Get AI-generated summaries and key insights from any YouTube video")
        
        youtube_video_url = st.text_input(
            "🔗 Enter YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=example",
            help="Paste the complete YouTube video URL here"
        )

        if youtube_video_url:
            try:
                with st.spinner("🔄 Extracting transcript..."):
                    transcript_text = extract_transcript_details(youtube_video_url)

                y_prompt = """
                You are a YouTube video summarizer. You will summarize the transcript text 
                and provide the key points within 250 words. Here is the transcript:
                """

                with st.spinner("🤖 Generating AI summary..."):
                    summary = generate_gemini_summary(transcript_text, y_prompt)

                if summary:
                    st.markdown("### 📝 **Video Summary:**")
                    st.info(summary)
                    
                    # Enhanced additional features
                    with st.expander("📊 Transcript Analytics", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Transcript Length", f"{len(transcript_text):,} characters")
                        with col2:
                            st.metric("Estimated Reading Time", f"{len(transcript_text.split()) // 200} minutes")
                else:
                    st.error("Failed to generate summary. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("💡 Make sure the YouTube URL is valid and the video has captions available.")

    elif option == "Chat with PDFs":
        st.markdown("### 📚 PDF Chat Assistant")
        st.markdown("Upload your PDF documents and ask intelligent questions to get instant answers")
        
        user_question = st.text_input(
            "💬 Ask a Question from the PDF files:",
            placeholder="e.g., What is the main topic discussed in the document?",
            help="Type your question about the uploaded PDF content"
        )

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.markdown("### 📁 Upload Documents")
            st.markdown("Upload your PDF files to start chatting")
            
            pdf_docs = st.file_uploader(
                "Choose PDF Files",
                accept_multiple_files=True,
                type=['pdf'],
                help="Select one or more PDF files to analyze"
            )

            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                if pdf_docs:
                    with st.spinner("🔄 Processing your PDFs..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.warning("⚠️ Uploaded PDFs contain no text. Please upload valid PDFs.")
                            return
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ PDFs processed successfully! You can now ask questions.")
                else:
                    st.warning("📋 Please upload at least one PDF file.")

    # Display navigation section after main content
    display_navigation_section()

    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p>Built with ❤️ using Streamlit and Google Gemini AI</p>
        <p><small>For the best experience, ensure a stable internet connection</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()