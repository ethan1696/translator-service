from src.translator import translate_content
import openai
from unittest.mock import patch, MagicMock
from .utils import eval_single_response_complete


def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert eval_single_response_complete((False, "This is a Chinese message"), (is_english, translated_content)) > 0.9

@patch('openai.OpenAI')
def test_missing_separator(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="True This is probably English but wrong format"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    assert translate_content("Bonjour") == (True, "Bonjour")

@patch('openai.OpenAI')
def test_gibberish_response(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="asdfasdf"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    assert translate_content("你好吗？") == (True, "你好吗？")

@patch('openai.OpenAI')
def test_invalid_boolean(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Maybe | This is maybe English"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    assert translate_content("¿Dónde está la biblioteca?") == (True, "¿Dónde está la biblioteca?")

@patch('openai.OpenAI')
def test_api_exception(mock_openai):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("LLM call failed")
    mock_openai.return_value = mock_client

    assert translate_content("Hello there!") == (True, "Hello there!")