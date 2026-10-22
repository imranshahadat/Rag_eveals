from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage

chat = ChatUpstage(
    api_key="up_XcGMbhqy2cwgX0Xcs81GhUUpgG6Qs",
    model="solar-mini",
)

response = chat.invoke([
    HumanMessage(
        content="An $80 item gets a 20% discount, then 10% tax is added. What is the final price? Answer with only the dollar amount."
    )
])

reasoning = response.additional_kwargs.get("reasoning_content")
if reasoning:
    print(reasoning)

print(response.content)