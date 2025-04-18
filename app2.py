import streamlit as st
import os
import sqlite3
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content([prompt[0], question])
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return None

def get_db_connection(sql, db):
    try:
        connection = sqlite3.connect(db)
        cur = connection.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        connection.commit()
        connection.close()
        return rows
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return []

def create_table_from_df(df):
    connection = sqlite3.connect('data.db')
    cursor = connection.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS data")
    
    sanitized_columns = [col.replace(" ", "_").replace(":", "_").replace("-", "_") for col in df.columns]
    
    columns_str = ", ".join([f'"{col}" TEXT' for col in sanitized_columns])
    create_table_query = f"CREATE TABLE IF NOT EXISTS data ({columns_str});"
    
    print(f"Executing SQL: {create_table_query}")
    cursor.execute(create_table_query)
    connection.commit()
    return cursor, connection

def insert_data_from_excel(file):
    try:
        # Read the Excel file
        df = pd.read_excel(file, header=0)
        cursor, connection = create_table_from_df(df)
        if not cursor:
            return False

        # Insert data into the newly created table
        for index, row in df.iterrows():
            placeholders = ", ".join(["?"] * len(row)) 
            sanitized_columns = [col.replace(" ", "_").replace(":", "_").replace("-", "_") for col in df.columns]
            column_names = ', '.join([f'"{col}"' for col in sanitized_columns])  
            insert_query = f"INSERT INTO data ({column_names}) VALUES ({placeholders})"
    
            print(f"Inserting SQL: {insert_query}, Values: {tuple(row)}")
    
            try:
                cursor.execute(insert_query, tuple(row))  
            except sqlite3.Error as e:
                print(f"Error inserting data: {e}") 

        connection.commit()
        connection.close()
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

# Streamlit UI
st.set_page_config(page_title="SQL Query Generator", page_icon="🔍")
st.header("Gemini App to Retrieve SQL Queries")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    with st.spinner("Inserting data into the database..."):
        success = insert_data_from_excel(uploaded_file)
        if success:
            st.success("Data inserted successfully!")
        else:
            st.error("Failed to insert data.")

question = st.text_area("Input:", key="input")

submit = st.button("Ask the question")

if submit:
    if not question.strip():
        st.error("Please enter a question.")
    else:
        prompt = [
            """
            You are an expert in converting English questions to SQL queries.
            The SQL database has a table named data with dynamic columns based on uploaded data.\n\n
            Example 1 - How many entries of records are present?
            SQL command: SELECT COUNT(*) FROM data;\n\n
            Example 2 - Sum up all values in the specified column.
            SQL command: SELECT SUM(<specified column>) FROM data;\n\n
            Ensure that your SQL commands do not include any formatting characters like ```
            """
        ]
        
        response = get_gemini_response(question, prompt)
        if response:
            sql_query = response.replace("SQL command:", "").strip()
            st.write(f"Generated SQL Query: {sql_query}")

            # Execute the SQL query
            data = get_db_connection(sql_query, "data.db")
            st.subheader("Query Results:")
            if data:
                st.dataframe(pd.DataFrame(data))
            else:
                st.write("No results found.")

