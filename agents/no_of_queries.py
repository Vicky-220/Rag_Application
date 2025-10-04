from ollama import chat
from pydantic import BaseModel

system_prompt = """you are a AI agent who helps to extract number queries from user input.
basically you need to see how many queries can be differentiated and taken out from the user raw input weather the user may not clearly saparate them but still meaningfully saparate the queries from the whole context and return then in a list  of todos of user requests.
Keep in mind that you have to add the keywords into the queries sometime from the previous converstion context if the user is not clearly mentioning them in the current input because later you have to pass these queries to another agent to work better on them"""

class user_queries(BaseModel):
    queries : list[str]

def parse_queries(query: str, chat_context):
    """parse user queries from raw input"""
    response = chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {chat_context}\n\nQuery: {query}"}
        ],
        model="qwen2.5:3b-instruct-q8_0",
        format=user_queries.model_json_schema()
    )
    list_queries = user_queries.model_validate_json(response['message']['content']).queries

    print(list_queries)
    return list_queries