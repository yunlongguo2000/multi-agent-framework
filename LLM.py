import openai
import tiktoken
import time

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

openai_api_key_name = 'sk-'
openai_base_url = 'https://api.chatanywhere.tech/v1'

client = openai.OpenAI(
    api_key=openai_api_key_name,
    base_url=openai_base_url
)

def GPT_response(messages, model_name):
    token_num_count = 0
    for item in messages:
        token_num_count += len(enc.encode(item["content"]))

    if model_name in ['gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613','gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo']:
        try:
            result = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        except:
            try:
                print(f'{model_name} Waiting 60 seconds for API query')
                time.sleep(60)
                result = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            except:
                return 'Out of tokens', token_num_count
        token_num_count += len(enc.encode(result.choices[0].message.content))
        print(f'Token_num_count: {token_num_count}')
        return result.choices[0].message.content, token_num_count

    else:
        raise ValueError(f'Invalid model name: {model_name}')