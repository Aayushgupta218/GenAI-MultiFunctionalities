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

def display_navigation_section():
    """Display navigation section with links to other applications"""
    st.markdown("---")
    st.markdown("## 🚀 **Explore More AI Tools**")
    st.markdown("### Access additional AI-powered applications:")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # SQL Query Generator Button
        st.markdown("#### 🔍 **SQL Query Generator**")
        st.markdown("*Generate SQL queries from natural language using AI*")
        
        sql_url = "https://sqlquerygenerator-7xbeqcgnivqtywjghc4sd3.streamlit.app/"
        
        # Custom styled button using HTML
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <a href="{sql_url}" target="_blank" style="text-decoration: none;">
                <button style="
                    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                    color: white;
                    padding: 15px 30px;
                    font-size: 16px;
                    font-weight: bold;
                    border: none;
                    border-radius: 25px;
                    cursor: pointer;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    transition: all 0.3s ease;
                    width: 250px;
                    height: 50px;
                " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 6px 20px rgba(0, 0, 0, 0.3)';" 
                   onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 4px 15px rgba(0, 0, 0, 0.2)';">
                    🔍 Open SQL Generator
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # Alternative method using st.link_button (if available in your Streamlit version)
        # st.link_button(
        #     "🔍 Open SQL Query Generator", 
        #     sql_url,
        #     help="Generate SQL queries from natural language",
        #     use_container_width=True
        # )
        
        # Features list
        with st.expander("✨ SQL Generator Features"):
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
    
    # Main title with improved styling
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #2E86AB; font-size: 2.5em; margin-bottom: 10px;">
            🧠 AI Document & Video Suite
        </h1>
        <p style="font-size: 1.2em; color: #666; margin-bottom: 30px;">
            Powered by Google Gemini AI - Your AI Assistant for Documents and Videos
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display navigation section at the top
    display_navigation_section()
    
    # Main application content
    st.markdown("---")
    st.markdown("## 📄 **Current Application Features**")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing Google API Key. Please set it in the .env file.")
        return

    option = st.selectbox(
        "Choose a task:", 
        ["Chat with PDFs", "Summarize YouTube Video"],
        help="Select the AI feature you want to use"
    )

    if option == "Chat with PDFs":
        st.markdown("### 📚 Chat with Multiple PDF Files")
        st.markdown("*Ask questions about your uploaded PDF documents*")
        
        user_question = st.text_input(
            "Ask a Question from the PDF files...",
            placeholder="e.g., What is the main topic discussed in the document?"
        )

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.markdown("### 📁 **Upload Documents**")
            pdf_docs = st.file_uploader(
                "Upload your PDF Files and Click Submit", 
                accept_multiple_files=True,
                type=['pdf'],
                help="Select one or more PDF files to analyze"
            )

            if st.button("🔄 Submit & Process", type="primary"):
                if pdf_docs:
                    with st.spinner("Processing your PDFs..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.warning("Uploaded PDFs contain no text. Please upload valid PDFs.")
                            return
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ PDFs processed successfully! You can now ask questions.")
                else:
                    st.warning("Please upload at least one PDF file.")

    elif option == "Summarize YouTube Video":
        st.markdown("### 🎥 Summarize YouTube Video")
        st.markdown("*Get AI-generated summaries of YouTube videos*")
        
        youtube_video_url = st.text_input(
            "Enter YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=example"
        )

        if youtube_video_url:
            try:
                with st.spinner("Extracting transcript..."):
                    transcript_text = extract_transcript_details(youtube_video_url)

                y_prompt = """
                You are a YouTube video summarizer. You will summarize the transcript text 
                and provide the key points within 250 words. Here is the transcript:
                """

                with st.spinner("Generating summary..."):
                    summary = generate_gemini_summary(transcript_text, y_prompt)

                if summary:
                    st.markdown("### 📝 **Video Summary:**")
                    st.write(summary)
                    
                    # Additional features
                    with st.expander("📊 Transcript Details"):
                        st.write(f"**Transcript Length:** {len(transcript_text)} characters")
                        st.write(f"**Estimated Reading Time:** {len(transcript_text.split()) // 200} minutes")
                else:
                    st.error("Failed to generate summary. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Make sure the YouTube URL is valid and the video has captions available.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p>Built with ❤️ using Streamlit and Google Gemini AI</p>
        <p><small>For the best experience, ensure a stable internet connection</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()