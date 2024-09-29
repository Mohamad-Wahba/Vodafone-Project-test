"""
Vodafone Chatbot Application

This application provides a chat interface for interacting with a MySQL database using natural language queries. 
It utilizes an open source LLM for query generation and response formatting.

The application is built with Streamlit for the frontend and includes real-time application monitoring functionality.
It also supports visualization of data using matplotlib.

Authors: Mohamed Bassiony, Mohamad Wahba, Beshoy Ashraf Samir, Mohamad Sharqawi
Date: September 24, 2024
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from sqlalchemy.exc import SQLAlchemyError
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import time
from monitoring import get_app_metrics

# Load environment variables
load_dotenv()

# Initialize session state for chat history and metrics
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello 👋! How can I assist you?"),
    ]
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "cpu_usage": 0,
        "memory_usage": 0,
        "db_queries": 0
    }

# Initialize the model
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

def init_database(
    user: str, password: str, host: str, port: str, database: str
) -> SQLDatabase:
    """
    Initialize and return a SQLDatabase object for the given MySQL database.
    """
    try:
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        return SQLDatabase.from_uri(db_uri)
    except mysql.connector.Error as err:
        if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
            st.error("Error: Invalid username or password. Please check your credentials.")
        elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
            st.error("Error: The specified database does not exist.")
        else:
            st.error(f"Error: Unable to connect to the database. Please check your connection settings. (Error: {err})")
        return None
    except SQLAlchemyError as e:
        st.error(f"An error occurred while connecting to the database: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def check_plotting():
    template = """
    You are a pandas ai expert at a company. You are interacting with a user who is asking you questions about the company's database.
    check that the user's question needs to be plotting. Take the conversation history into account.
    
    Write only the Boolean Response and nothing else. Do not wrap the Boolean Response in any other text, not even backticks or space.
    
    For example:
    Question: Plot a histogram of countries showing GDP, using different colors for each bar.
    Boolean Response:True
    Question: Create a line chart of sales over time?
    Boolean Response:True
    Question: How many employees are there?
    Boolean Response:False
    
    Your turn:
    
    Question: {question}
    Boolean Response:
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

def get_sql_chain(db):
    """
    Create and return a chain for generating SQL queries based on user questions.
    """
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    return RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser()

def get_response(user_query: str, db: SQLDatabase, chat_history: list) -> str:
    """
    Generate a response to the user's query using the database and chat history.
    """
    try:
        sql_chain = get_sql_chain(db)

        template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]),
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        st.session_state.metrics["db_queries"] += 1
        return chain.invoke(
            {
                "question": user_query,
                "chat_history": chat_history,
            }
        )
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your request. Please try rephrasing your question or contact the IT support team if the issue persists. (Error details: {str(e)})"

def get_data_for_plotting(db, query):
    try:
        st.session_state.metrics["db_queries"] += 1
        result = db.run(query)
        return pd.DataFrame(result)
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def plot_data(df, x_column, y_column, title):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(df[x_column], df[y_column])
        plt.title(title)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_path = "temp_plot.png"
        plt.savefig(img_path)
        plt.close()
        
        return img_path
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

# Streamlit app
def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Vodafone chatbot", page_icon=":speech_balloon:")
    st.title("Chat with Vodafone database")

    # Initialize session state for database connection
    if "db" not in st.session_state:
        st.session_state.db = None

    # Sidebar for database connection settings and application monitoring
    with st.sidebar:
        st.image("https://1000logos.net/wp-content/uploads/2017/06/Vodafone_Logo.png")
        st.subheader("Settings")
        st.write("Connect Vodafone local database and start chatting.")

        st.text_input("Host", value=os.getenv("DB_HOST", "localhost"), key="Host")
        st.text_input("Port", value=os.getenv("DB_PORT", "3306"), key="Port")
        st.text_input("User", value=os.getenv("DB_USER", "root"), key="User")
        st.text_input("Password", type="password", value=os.getenv("DB_PASSWORD", ""), key="Password")
        st.text_input("Database", value=os.getenv("DB_NAME", "Chinook"), key="Database")

        if st.button("Connect"):
            with st.spinner("Connecting to database..."):
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"],
                )
                if db:
                    st.session_state.db = db
                    st.success("Connected to database successfully!")
                else:
                    st.error("Failed to connect to the database. Please check your settings and try again.")

        # Application Monitoring Section
        st.subheader("Application Monitoring")
        metrics_placeholder = st.empty()

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJ9uo29rKi1lXepifVFHiXtetcFLN7dyZhcQ&s"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    # Handle user input
    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        if st.session_state.db is None:
            st.error("Please connect to the database first before asking questions.")
        else:
            with st.chat_message("AI", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJ9uo29rKi1lXepifVFHiXtetcFLN7dyZhcQ&s"):
                with st.spinner("Generating response..."):
                    try:
                        check_plotting_chain = check_plotting()
                        boolean_plotting = check_plotting_chain.invoke({"question": user_query})

                        if boolean_plotting.lower() == "true":
                            sql_chain = get_sql_chain(st.session_state.db)
                            query = sql_chain.invoke({
                                "chat_history": st.session_state.chat_history,
                                "question": user_query,
                            })
                            
                            df = get_data_for_plotting(st.session_state.db, query)
                            
                            if df is not None and not df.empty:
                                x_column = df.columns[0]
                                y_column = df.columns[1]
                                
                                img_path = plot_data(df, x_column, y_column, user_query)
                                
                                if img_path:
                                    st.image(img_path)
                                    with open(img_path, "rb") as file:
                                        btn = st.download_button(
                                            label="Download image",
                                            data=file,
                                            file_name="plot.png",
                                            mime="image/png"
                                        )
                                    os.remove(img_path)  # Clean up the temporary file
                                    response = f"Here's the plot you requested for '{user_query}'. The x-axis represents {x_column} and the y-axis represents {y_column}."
                                else:
                                    response = "I apologize, but I encountered an error while creating the plot. Please try rephrasing your question or contact the IT support team if the issue persists."
                            else:
                                response = "I apologize, but I couldn't retrieve the necessary data for plotting. Please check if the data exists in the database or try rephrasing your question."
                        else:
                            response = get_response(
                                user_query,
                                st.session_state.db,
                                st.session_state.chat_history,
                            )
                        
                        st.markdown(response)
                        st.session_state.chat_history.append(AIMessage(content=response))
                    except Exception as e:
                        error_message = f"I apologize, but I encountered an unexpected error while processing your request. Please try again later or contact the IT support team if the issue persists. (Error details: {str(e)})"
                        st.error(error_message)
                        st.session_state.chat_history.append(AIMessage(content=error_message))

    # Update metrics display
    metrics_placeholder.empty()
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        metrics = get_app_metrics()
        col1.metric("CPU Usage", f"{metrics['cpu_usage']:.2f}%")
        col2.metric("Memory Usage", f"{metrics['memory_usage']:.2f} MB")
        col3.metric("DB Queries", st.session_state.metrics['db_queries'])

if __name__ == "__main__":
    main()
