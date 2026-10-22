import os
import requests

response = requests.post(
    'https://api.apodex.ai/v1/chat/completions',
    headers={
        'Authorization': f'Bearer sk-NEA_t0TAxcL1CY5FYZ6u9YE-sr3Cy8kMHHlqnH6i-qc=',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'apodex-1.1',
        'messages': [
            {'role': 'user', 'content': 'say hello worlf too me'}
        ],
        'stream': False,                  
        'reasoning_effort': 'none'       
    }
)

# Parse the final JSON response directly
data = response.json()

# Print only the clean, final answer text
if 'choices' in data:
    print(data['choices'][0]['message']['content'])
else:
    print("Error:", data)
