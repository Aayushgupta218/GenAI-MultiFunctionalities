import streamlit as st
import os
import sqlite3
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Initialize environment and API
load_dotenv()

def check_api_key():
    """Check if Google API key is available"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔑 Google API Key not found!")
        st.stop()
    return api_key

def initialize_gemini():
    try:
        api_key = check_api_key()
        genai.configure(api_key=api_key)
        
        # Test the connection
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        st.stop()

def get_gemini_response(question, prompt, model):
    """Get response from Gemini API with proper error handling"""
    try:
        # Combine prompt and question properly
        full_prompt = f"{prompt}\n\nQuestion: {question}"
        
        # Generate response with safety settings
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            st.error("Empty response from Gemini API")
            return None
            
    except genai.types.BlockedPromptException:
        st.error("Prompt was blocked by safety filters. Try rephrasing your question.")
        return None
    except genai.types.StopCandidateException:
        st.error("Response was stopped due to safety reasons.")
        return None
    except Exception as e:
        st.error(f"Error with Gemini API: {str(e)}")
        st.info("Please check your API key and internet connection")
        return None

def sanitize_column_name(col_name):
    """Sanitize column names for SQL compatibility"""
    # Convert to string and handle NaN/None values
    if pd.isna(col_name):
        col_name = "unnamed_column"
    
    col_name = str(col_name).strip()
    
    # Replace problematic characters
    sanitized = re.sub(r'[^\w\s]', '_', col_name)
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = f"col_{sanitized}"
    
    # Handle empty names
    if not sanitized:
        sanitized = "unnamed_column"
    
    return sanitized

def get_db_connection(sql, db_name="data.db"):
    """Execute SQL query and return results with proper error handling"""
    try:
        if not os.path.exists(db_name):
            st.error("Database file not found. Please upload data first.")
            return []
        
        connection = sqlite3.connect(db_name)
        cursor = connection.cursor()
        
        # Execute the query
        cursor.execute(sql)
        
        # Fetch results
        if sql.strip().upper().startswith(('SELECT', 'WITH')):
            rows = cursor.fetchall()
            # Get column names for better display
            column_names = [description[0] for description in cursor.description]
            connection.close()
            return rows, column_names
        else:
            connection.commit()
            connection.close()
            return [], []
            
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return [], []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return [], []

def create_table_from_df(df):
    """Create SQLite table from DataFrame with proper error handling"""
    try:
        connection = sqlite3.connect('data.db')
        cursor = connection.cursor()
        
        # Drop existing table
        cursor.execute("DROP TABLE IF EXISTS data")
        
        # Sanitize column names
        original_columns = df.columns.tolist()
        sanitized_columns = [sanitize_column_name(col) for col in original_columns]
        
        # Create column mapping for reference
        column_mapping = dict(zip(original_columns, sanitized_columns))
        
        # Ensure unique column names
        seen_columns = set()
        final_columns = []
        for col in sanitized_columns:
            if col in seen_columns:
                counter = 1
                while f"{col}_{counter}" in seen_columns:
                    counter += 1
                col = f"{col}_{counter}"
            seen_columns.add(col)
            final_columns.append(col)
        
        # Create table with sanitized column names
        columns_str = ", ".join([f'"{col}" TEXT' for col in final_columns])
        create_table_query = f"CREATE TABLE data ({columns_str});"
        
        cursor.execute(create_table_query)
        connection.commit()
        
        return cursor, connection, final_columns, column_mapping
        
    except Exception as e:
        st.error(f"Error creating table: {e}")
        return None, None, None, None

def insert_data_from_excel(file):
    """Insert data from Excel file into SQLite database"""
    try:
        # Read the Excel file with proper error handling
        try:
            df = pd.read_excel(file, header=0)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return False, None
        
        if df.empty:
            st.error("The uploaded file is empty")
            return False, None
        
        # Create table
        cursor, connection, final_columns, column_mapping = create_table_from_df(df)
        if not cursor:
            return False, None
        
        # Prepare data for insertion
        df_renamed = df.copy()
        df_renamed.columns = final_columns
        
        # Convert all data to strings and handle NaN values
        df_renamed = df_renamed.fillna('')
        df_renamed = df_renamed.astype(str)
        
        # Insert data
        placeholders = ", ".join(["?"] * len(final_columns))
        column_names = ', '.join([f'"{col}"' for col in final_columns])
        insert_query = f"INSERT INTO data ({column_names}) VALUES ({placeholders})"
        
        # Insert data row by row with error handling
        successful_inserts = 0
        for index, row in df_renamed.iterrows():
            try:
                cursor.execute(insert_query, tuple(row))
                successful_inserts += 1
            except sqlite3.Error as e:
                st.warning(f"Error inserting row {index + 1}: {e}")
                continue
        
        connection.commit()
        connection.close()
        
        st.info(f"Successfully inserted {successful_inserts} out of {len(df)} rows")
        return True, column_mapping
        
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return False, None

def clean_sql_response(response):
    """Clean the SQL response from Gemini"""
    if not response:
        return ""
    
    # Remove markdown code blocks
    response = re.sub(r'```sql\n?', '', response, flags=re.IGNORECASE)
    response = re.sub(r'```\n?', '', response)
    
    # Remove common prefixes
    response = re.sub(r'^(SQL command:|SQL query:|Query:)\s*', '', response, flags=re.IGNORECASE)
    
    # Extract the actual SQL query (first complete statement)
    lines = response.strip().split('\n')
    sql_query = ""
    for line in lines:
        line = line.strip()
        if line and not line.startswith('--') and not line.startswith('#'):
            sql_query += line + " "
            if line.endswith(';'):
                break
    
    return sql_query.strip()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="SQL Query Generator", 
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 SQL Query Generator with Gemini AI")
    st.markdown("Upload an Excel file and ask questions about your data in natural language!")
    
    # Initialize Gemini
    model = initialize_gemini()
    
    # File upload section
    st.subheader("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose an Excel file", 
        type=["xlsx", "xls"],
        help="Upload an Excel file to create a database table"
    )
    
    column_mapping = None
    
    if uploaded_file is not None:
        with st.spinner("Processing your data..."):
            success, column_mapping = insert_data_from_excel(uploaded_file)
            if success:
                st.success("✅ Data uploaded successfully!")
                
                # Show column mapping if columns were renamed
                if column_mapping:
                    with st.expander("📋 Column Mapping (Original → Database)"):
                        mapping_df = pd.DataFrame(
                            list(column_mapping.items()), 
                            columns=['Original Column', 'Database Column']
                        )
                        st.dataframe(mapping_df, use_container_width=True)
            else:
                st.error("❌ Failed to upload data. Please check your file format.")
                return
    
    # Query section
    st.subheader("❓ Ask Your Question")
    question = st.text_area(
        "Enter your question about the data:",
        height=100,
        placeholder="Example: How many records are there? What is the average of column X?",
        key="input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("🚀 Generate Query", type="primary")
    
    if submit:
        if not question.strip():
            st.error("Please enter a question.")
        elif not os.path.exists('data.db'):
            st.error("Please upload data first.")
        else:
            # Create enhanced prompt
            prompt = """
            You are an expert in converting English questions to SQL queries.
            The SQL database has a table named 'data' with columns based on the uploaded Excel file.
            
            Rules:
            1. Always use the table name 'data'
            2. Use proper SQL syntax
            3. For counting records, use: SELECT COUNT(*) FROM data;
            4. For column operations, make sure column names exist
            5. Use double quotes around column names if they contain spaces or special characters
            6. Return only the SQL query, no explanations or formatting
            7. End queries with semicolon
            
            Examples:
            - "How many records are there?" → SELECT COUNT(*) FROM data;
            - "Show all data" → SELECT * FROM data;
            - "What is the sum of column X?" → SELECT SUM("column_X") FROM data;
            """
            
            with st.spinner("Generating SQL query..."):
                response = get_gemini_response(question, prompt, model)
                
                if response:
                    # Clean the response
                    sql_query = clean_sql_response(response)
                    
                    if sql_query:
                        st.subheader("🔧 Generated SQL Query")
                        st.code(sql_query, language="sql")
                        
                        # Execute the query
                        with st.spinner("Executing query..."):
                            try:
                                results, column_names = get_db_connection(sql_query)
                                
                                st.subheader("📊 Query Results")
                                if results:
                                    if column_names:
                                        # Create DataFrame with column names
                                        df_results = pd.DataFrame(results, columns=column_names)
                                        st.dataframe(df_results, use_container_width=True)
                                        
                                        # Show summary
                                        st.info(f"Found {len(results)} record(s)")
                                    else:
                                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                                else:
                                    st.info("No results found for your query.")
                                    
                            except Exception as e:
                                st.error(f"Error executing query: {e}")
                                st.info("Try rephrasing your question or check if the column names are correct.")
                    else:
                        st.error("Could not generate a valid SQL query. Please try rephrasing your question.")
                else:
                    st.error("Failed to generate SQL query. Please try again.")
    
    # Add helpful information
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        - **Be specific**: Instead of "show data", try "show first 10 rows"
        - **Use column names**: Reference actual column names from your data
        - **Ask for counts**: "How many records have X condition?"
        - **Request aggregations**: "What is the average/sum/max of column Y?"
        - **Filter data**: "Show records where column Z equals 'value'"
        """)

if __name__ == "__main__":
    main()