"""
Vodafone Chatbot Application Tests

This module provides comprehensive unit tests for the Vodafone Chatbot Application.
It covers database operations, SQL chain generation, response handling, and Streamlit components.

Usage:
    Run these tests using pytest:
    $ pytest test_app.py

Authors: Mohamed Bassiony, Mohamad Wahba, Beshoy Ashraf Samir, Mohamad Sharqawi
Date: July 14, 2024
"""

import pytest
from unittest.mock import Mock, patch
from app import init_database, get_sql_chain, get_response, main

@pytest.fixture
def mock_db():
    return Mock()

@pytest.fixture
def mock_llm():
    return Mock()

def test_init_database():
    with patch("app.SQLDatabase") as mock_sql_database:
        mock_sql_database.from_uri.return_value = Mock()
        db = init_database("user", "password", "host", "port", "database")
        assert db is not None
        mock_sql_database.from_uri.assert_called_once_with(
            "mysql+mysqlconnector://user:password@host:port/database"
        )

def test_get_sql_chain(mock_db):
    with patch("app.ChatGroq") as mock_chat_groq, \
         patch("app.ChatPromptTemplate") as mock_chat_prompt_template:
        chain = get_sql_chain(mock_db)
        assert chain is not None
        mock_chat_groq.assert_called_once()
        mock_chat_prompt_template.assert_called_once()

@pytest.mark.asyncio
async def test_get_response(mock_db, mock_llm):
    with patch("app.get_sql_chain") as mock_get_sql_chain, \
         patch("app.ChatGroq", return_value=mock_llm):
        mock_get_sql_chain.return_value = Mock()
        mock_db.run.return_value = "Mock SQL Response"
        mock_db.get_table_info.return_value = "Mock Schema"
        mock_llm.invoke.return_value = "AI response"

        response = await get_response("What are the top selling products?", mock_db, [])
        assert isinstance(response, str)
        assert response == "AI response"
        mock_get_sql_chain.assert_called_once()
        mock_db.run.assert_called_once()
        mock_db.get_table_info.assert_called_once()

def test_streamlit_components():
    with patch("streamlit.set_page_config") as mock_set_page_config, \
         patch("streamlit.title") as mock_title, \
         patch("streamlit.sidebar.image") as mock_sidebar_image, \
         patch("streamlit.sidebar.subheader") as mock_sidebar_subheader, \
         patch("streamlit.sidebar.text_input") as mock_text_input, \
         patch("streamlit.sidebar.button") as mock_button, \
         patch("streamlit.chat_input") as mock_chat_input, \
         patch("streamlit.session_state", {"chat_history": []}):

        mock_text_input.side_effect = ["localhost", "3306", "root", "password", "Chinook"]
        mock_button.return_value = True
        mock_chat_input.return_value = "What are the top 5 selling albums?"

        main()

        mock_set_page_config.assert_called_once()
        mock_title.assert_called_once_with("Chat with Vodafone database")
        mock_sidebar_image.assert_called_once()
        assert mock_sidebar_subheader.call_count == 2
        assert mock_text_input.call_count == 5
        mock_button.assert_called_once_with("Connect")
        mock_chat_input.assert_called_once_with("Type a message...")

if __name__ == "__main__":
    pytest.main()
