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
from youtube_transcript_api.formatters import TextFormatter
import webbrowser
import re
from datetime import datetime
import json
import requests
import time
from urllib.parse import urlparse, parse_qs

load_dotenv()  # load all the env variables 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=300)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Enhanced prompt for more detailed and contextual responses
    prompt_template = """
    You are an advanced AI document analyst with expertise in comprehensive information extraction and analysis.
    
    INSTRUCTIONS:
    1. Analyze the provided context thoroughly and extract the most relevant information
    2. Provide detailed, well-structured answers with specific examples and evidence
    3. Use bullet points, numbered lists, and clear formatting for better readability
    4. If information spans multiple sections, synthesize and connect the concepts
    5. Include relevant quotes or specific data points when available
    6. If the answer requires inference from the context, clearly indicate your reasoning
    7. For complex topics, break down the answer into logical sections
    8. If the context doesn't contain the answer, suggest related topics that are covered
    
    CONTEXT:
    {context}
    
    QUESTION: {question}
    
    DETAILED ANALYSIS:
    """
    # Using gemini-1.5-flash for better free tier performance
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def extract_transcript_details(youtube_video_url):
    """Enhanced transcript extraction with multiple fallback methods"""
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            raise Exception("Invalid YouTube URL format")
        
        # Method 1: Try with different language codes
        language_codes = ['en', 'en-US', 'en-GB', 'auto']
        transcript_text = None
        formatted_transcript = []
        
        for lang in language_codes:
            try:
                if lang == 'auto':
                    # Try to get any available transcript
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    for transcript in transcript_list:
                        try:
                            transcript_data = transcript.fetch()
                            break
                        except:
                            continue
                else:
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                
                # If we got transcript data, process it
                if transcript_data:
                    full_text = ""
                    for item in transcript_data:
                        timestamp = f"[{int(item['start']//60):02d}:{int(item['start']%60):02d}]"
                        formatted_transcript.append(f"{timestamp} {item['text']}")
                        full_text += item["text"] + " "
                    
                    return full_text, formatted_transcript
                    
            except Exception as e:
                continue
        
        # Method 2: Try manual transcript extraction (if available)
        if not transcript_text:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                # Get the first available transcript
                transcript = next(iter(transcript_list))
                transcript_data = transcript.fetch()
                
                full_text = ""
                for item in transcript_data:
                    timestamp = f"[{int(item['start']//60):02d}:{int(item['start']%60):02d}]"
                    formatted_transcript.append(f"{timestamp} {item['text']}")
                    full_text += item["text"] + " "
                
                return full_text, formatted_transcript
                
            except Exception as e:
                pass
        
        # If all methods fail, provide alternative
        raise Exception("No transcript available - try a different video or check if captions are enabled")
        
    except Exception as e:
        error_msg = str(e)
        if "blocked" in error_msg.lower() or "ip" in error_msg.lower():
            raise Exception("""
            🚫 YouTube API Access Blocked
            
            This happens when:
            • Too many requests from your IP
            • Using cloud/server IP (like Streamlit Cloud)
            • Regional restrictions
            
            💡 Solutions:
            1. Try a different video with captions
            2. Wait 10-15 minutes and try again
            3. Use a video you know has auto-generated captions
            4. Try running locally instead of on cloud platforms
            
            📝 Alternative: Copy and paste the transcript manually if available
            """)
        else:
            raise Exception(f"Transcript extraction failed: {error_msg}")

def manual_transcript_input():
    """Allow users to manually input transcript if API fails"""
    st.markdown("### 📝 **Manual Transcript Input**")
    st.info("💡 If API fails, you can manually copy-paste the transcript from YouTube")
    
    with st.expander("📋 How to get transcript manually", expanded=False):
        st.markdown("""
        **Steps to get transcript manually:**
        1. Go to your YouTube video
        2. Click on "..." (More) below the video
        3. Click "Show transcript"
        4. Copy all the text and paste it below
        5. Or use YouTube's auto-generated captions
        """)
    
    manual_transcript = st.text_area(
        "Paste transcript here:",
        placeholder="Paste the YouTube video transcript here...",
        height=200,
        help="Copy the transcript from YouTube and paste it here"
    )
    
    if manual_transcript and len(manual_transcript.strip()) > 50:
        # Process manual transcript
        formatted_transcript = manual_transcript.split('\n')
        return manual_transcript, formatted_transcript
    
    return None, None

def generate_comprehensive_summary(transcript_text, summary_type="detailed"):
    """Generate different types of summaries based on user preference"""
    
    prompts = {
        "detailed": """
        You are a professional content analyst and summarization expert. Analyze this video transcript and provide a comprehensive, structured summary.
        
        ANALYSIS FRAMEWORK:
        1. **Main Topic & Thesis**: Identify the core subject and primary argument/message
        2. **Key Points & Arguments**: Extract 5-7 most important points with supporting details
        3. **Evidence & Examples**: List specific examples, data, or case studies mentioned
        4. **Conclusions & Takeaways**: Summarize main conclusions and actionable insights
        5. **Context & Background**: Provide relevant background information discussed
        6. **Target Audience**: Identify who this content is aimed at
        7. **Content Quality Assessment**: Brief evaluation of the content's depth and reliability
        
        FORMAT YOUR RESPONSE AS:
        ## 🎯 **Main Topic & Thesis**
        [Clear statement of what the video is about and its primary message]
        
        ## 📋 **Key Points & Arguments**
        1. **[Point 1 Title]**: [Detailed explanation with context]
        2. **[Point 2 Title]**: [Detailed explanation with context]
        [Continue for all major points]
        
        ## 🔍 **Evidence & Examples**
        • [Specific examples, statistics, or case studies mentioned]
        
        ## 💡 **Key Takeaways & Conclusions**
        • [Actionable insights and main conclusions]
        
        ## 🎯 **Target Audience & Application**
        [Who should watch this and how they can apply the information]
        
        TRANSCRIPT TO ANALYZE:
        """,
        
        "executive": """
        You are a senior executive assistant creating a briefing summary. Provide a concise yet comprehensive executive summary focusing on:
        
        **EXECUTIVE BRIEFING STRUCTURE:**
        1. **Strategic Overview** (2-3 sentences)
        2. **Key Business Insights** (3-5 bullet points)
        3. **Action Items & Opportunities** (2-4 points)
        4. **Risks & Considerations** (if applicable)
        5. **Recommendation** (1-2 sentences)
        
        Keep it professional, data-driven, and decision-focused. Limit to 200-250 words.
        
        TRANSCRIPT:
        """,
        
        "academic": """
        You are an academic researcher creating a scholarly summary. Provide:
        
        **ACADEMIC ANALYSIS:**
        1. **Research Question/Hypothesis** addressed
        2. **Methodology** or approach discussed
        3. **Key Findings** with evidence
        4. **Theoretical Framework** or concepts
        5. **Limitations** or gaps identified
        6. **Future Research Directions**
        7. **Citations & References** mentioned
        
        Use academic tone and structure. Include critical analysis where appropriate.
        
        TRANSCRIPT:
        """,
        
        "technical": """
        You are a technical documentation specialist. Create a structured technical summary:
        
        **TECHNICAL BREAKDOWN:**
        1. **Technical Scope** & domain covered
        2. **Core Technologies/Concepts** explained
        3. **Implementation Details** discussed
        4. **System Architecture** or design patterns
        5. **Best Practices** & recommendations
        6. **Common Issues** & solutions
        7. **Prerequisites** & skill requirements
        
        Focus on technical accuracy and practical application.
        
        TRANSCRIPT:
        """
    }
    
    try:
        # Use gemini-1.5-flash for better free tier performance
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        if not transcript_text:
            return "Error: No transcript available"
            
        selected_prompt = prompts.get(summary_type, prompts["detailed"])
        response = model.generate_content(selected_prompt + transcript_text)
        
        if hasattr(response, 'text'):
            return response.text
        return "Sorry, couldn't generate summary."
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        st.info("💡 If you hit rate limits, try again in a moment or use fewer requests")
        return None

def generate_smart_questions(content_text, content_type="pdf"):
    """Generate intelligent questions based on content analysis"""
    prompt = f"""
    Analyze this {'PDF document' if content_type == 'pdf' else 'video transcript'} and generate 8-10 intelligent questions that would help users explore the content deeper.
    
    QUESTION CATEGORIES:
    1. **Factual Questions** (2-3): Direct information retrieval
    2. **Analytical Questions** (2-3): Require analysis and synthesis  
    3. **Application Questions** (2-3): How to apply the information
    4. **Critical Thinking** (1-2): Evaluate or critique the content
    
    Format as: **Category**: Question text
    
    CONTENT TO ANALYZE:
    {content_text[:3000]}...
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else "Could not generate questions"
    except:
        return "Error generating questions"

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=4)  # Get more relevant chunks
        
        if not docs:
            st.warning("No relevant information found in the uploaded PDFs for the given question.")
            return
            
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        # Enhanced response display
        st.markdown("### 🤖 **AI Analysis:**")
        st.markdown(response["output_text"])
        
        # Show source relevance
        with st.expander("📚 **Source Context Used**", expanded=False):
            for i, doc in enumerate(docs[:2], 1):
                st.markdown(f"**Source {i}:**")
                st.text(doc.page_content[:300] + "...")
                
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
    """Display enhanced feature cards"""
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
            <h3 style="margin: 0 0 10px 0; font-size: 1.3em;">Advanced YouTube Analyzer</h3>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">Multi-format summaries, smart questions, and deep content analysis with timestamp tracking</p>
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
            <h3 style="margin: 0 0 10px 0; font-size: 1.3em;">Intelligent PDF Assistant</h3>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">Advanced document analysis with contextual responses, smart suggestions, and comprehensive insights</p>
        </div>
        """, unsafe_allow_html=True)

def display_navigation_section():
    """Display navigation section with links to other applications"""
    st.markdown("---")
    
    st.markdown("## 🚀 Explore More AI Tools")
    st.markdown("Discover additional AI-powered applications to boost your productivity")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
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
        
        sql_url = "https://sqlquerygenerator-7xbeqcgnivqtywjghc4sd3.streamlit.app/"
        
        if st.button("🚀 Launch SQL Generator", key="sql_generator_btn", use_container_width=True):
            st.markdown(f'<meta http-equiv="refresh" content="0; url={sql_url}">', unsafe_allow_html=True)
            st.success("Redirecting to SQL Generator...")
        
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
        page_title="Advanced AI Document & Video Suite", 
        layout="wide", 
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced main title
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        color: white;
    ">
        <h1 style="font-size: 3em; margin-bottom: 15px; font-weight: 700;">
            🧠 Advanced AI Document & Video Suite
        </h1>
        <p style="font-size: 1.3em; opacity: 0.9; margin: 0;">
            Powered by Google Gemini Pro - Your Intelligent Assistant for Deep Content Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    display_feature_cards()

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing Google API Key. Please set it in the .env file.")
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        option = st.selectbox(
            "🎯 Select AI Tool:",
            ["Advanced YouTube Analyzer", "Intelligent PDF Assistant"],
            index=0,
            help="Select the AI feature you want to use"
        )

    st.markdown("---")

    if option == "Advanced YouTube Analyzer":
        st.markdown("### 🎥 Advanced YouTube Video Analyzer")
        st.markdown("Get comprehensive AI analysis with multiple summary formats and intelligent insights")
        
        youtube_video_url = st.text_input(
            "🔗 Enter YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=example",
            help="Paste the complete YouTube video URL here"
        )

        if youtube_video_url:
            # First try automatic extraction
            transcript_text = None
            formatted_transcript = None
            
            try:
                with st.spinner("🔄 Extracting transcript..."):
                    transcript_text, formatted_transcript = extract_transcript_details(youtube_video_url)
                    
            except Exception as e:
                st.error(f"❌ **Transcript Extraction Failed:**")
                st.error(str(e))
                
                # Show manual input option
                st.markdown("---")
                transcript_text, formatted_transcript = manual_transcript_input()
            
            # Process if we have transcript (either automatic or manual)
            if transcript_text and len(transcript_text.strip()) > 50:
                # Summary type selector
                col1, col2 = st.columns([1, 1])
                with col1:
                    summary_type = st.selectbox(
                        "📊 Choose Summary Type:",
                        ["detailed", "executive", "academic", "technical"],
                        help="Select the type of analysis that best fits your needs"
                    )

                with st.spinner("🤖 Generating comprehensive analysis..."):
                    summary = generate_comprehensive_summary(transcript_text, summary_type)

                if summary:
                    st.markdown("### 📋 **Comprehensive Analysis:**")
                    st.markdown(summary)
                    
                    # Additional features
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🤔 Generate Smart Questions", type="secondary"):
                            with st.spinner("Generating intelligent questions..."):
                                questions = generate_smart_questions(transcript_text, "video")
                                st.markdown("### 🤔 **Suggested Questions:**")
                                st.markdown(questions)
                    
                    with col2:
                        with st.expander("📊 **Content Analytics**", expanded=False):
                            words = transcript_text.split()
                            sentences = transcript_text.split('.')
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Word Count", f"{len(words):,}")
                                st.metric("Sentences", len(sentences))
                            with col_b:
                                st.metric("Reading Time", f"{len(words) // 200} min")
                                st.metric("Speaking Time", f"{len(words) // 150} min")
                    
                    # Transcript with timestamps (if available)
                    if formatted_transcript:
                        with st.expander("📝 **Full Transcript with Timestamps**", expanded=False):
                            st.text_area("Transcript", "\n".join(formatted_transcript[:50]), height=300)
                else:
                    st.error("Failed to generate analysis. Please try again.")
            elif youtube_video_url:
                st.warning("⚠️ Please provide a valid transcript (minimum 50 characters)")
        
        # Alternative video suggestions
        st.markdown("---")
        st.markdown("### 💡 **Suggested Videos for Testing:**")
        test_videos = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"
        ]
        
        col1, col2, col3 = st.columns(3)
        for i, video in enumerate(test_videos):
            with [col1, col2, col3][i]:
                if st.button(f"📺 Test Video {i+1}", key=f"test_video_{i}"):
                    st.session_state.test_video = video

    elif option == "Intelligent PDF Assistant":
        st.markdown("### 📚 Intelligent PDF Assistant")
        st.markdown("Advanced document analysis with contextual understanding and smart insights")
        
        # Enhanced question input with suggestions
        user_question = st.text_input(
            "💬 Ask an Intelligent Question:",
            placeholder="e.g., What are the key findings and their implications?",
            help="Ask detailed questions for comprehensive analysis"
        )

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.markdown("### 📁 Document Upload & Analysis")
            
            pdf_docs = st.file_uploader(
                "Upload PDF Documents",
                accept_multiple_files=True,
                type=['pdf'],
                help="Upload multiple PDFs for comprehensive analysis"
            )

            if st.button("🚀 Process & Analyze Documents", type="primary", use_container_width=True):
                if pdf_docs:
                    with st.spinner("🔄 Processing and analyzing documents..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.warning("⚠️ No text found in uploaded PDFs.")
                            return
                        
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ Documents processed successfully!")
                        
                        # Auto-generate smart questions
                        with st.spinner("Generating intelligent questions..."):
                            smart_questions = generate_smart_questions(raw_text[:4000], "pdf")
                            
                        st.markdown("### 🤔 **Suggested Questions:**")
                        st.markdown(smart_questions)
                        
                        # Document statistics
                        st.markdown("### 📊 **Document Statistics:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Characters", f"{len(raw_text):,}")
                            st.metric("Documents Processed", len(pdf_docs))
                        with col2:
                            st.metric("Text Chunks Created", len(text_chunks))
                            st.metric("Estimated Pages", len(raw_text) // 2000)
                else:
                    st.warning("📋 Please upload at least one PDF file.")
            
            # Quick question suggestions
            st.markdown("### 💡 **Quick Questions:**")
            quick_questions = [
                "What are the main topics covered?",
                "Summarize the key findings",
                "What are the practical applications?",
                "Identify any limitations or challenges",
                "What conclusions can be drawn?"
            ]
            
            for i, q in enumerate(quick_questions):
                if st.button(f"❓ {q}", key=f"quick_q_{i}"):
                    st.session_state.quick_question = q
                    user_input(q)

    display_navigation_section()

    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p>🚀 <strong>Advanced AI Suite</strong> - Built with ❤️ using Streamlit and Google Gemini Pro</p>
        <p><small>💡 Pro tip: Use specific, detailed questions for better AI analysis</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()