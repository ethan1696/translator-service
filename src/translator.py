import os
import openai

def translate_content(content: str) -> tuple[bool, str]:
    try:
        api_key = os.environ.get('API_KEY')
        client = openai.OpenAI(
            api_key=api_key
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for evaluating forum posts. "
                        "Your task is to determine whether a post is written in English or not. "
                        "If it is written in English, respond with:\n"
                        "True | <the original post>\n"
                        "If it is not in English but intelligible, respond with:\n"
                        "False | <the English translation>\n"
                        "If the post is unintelligible or malformed and no meaningful translation is possible, respond with:\n"
                        "True | Unintelligible input.\n"
                        "Only return a single line in that format: a boolean, a pipe character, and the appropriate message."
                    )
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        raw_output = response.choices[0].message.content.strip()

        if "|" not in raw_output:
            raise ValueError("Missing expected separator '|'.")

        is_english_str, message = raw_output.split("|", 1)
        is_english_str = is_english_str.strip().lower()
        message = message.strip()

        if is_english_str not in ["true", "false"]:
            raise ValueError("First token is not 'true' or 'false'.")

        is_english = is_english_str == "true"
        return is_english, message

    except Exception as e:
        return True, content
