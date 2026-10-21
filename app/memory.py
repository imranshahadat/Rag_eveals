from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langsmith import traceable
from app.logger import log_msg

class ChatMemoryManager:
    def __init__(self, llm, max_tokens: int = 1000, summary_trigger: int = 10, keep_last_n: int = 4, strategy: str = "last"):
        self.llm = llm
        self.max_tokens = max_tokens
        self.summary_trigger = summary_trigger
        self.keep_last_n = keep_last_n
        self.strategy = strategy
        self.chat_history: list[BaseMessage] = []
        self.running_summary: str = ""

    def add_user_message(self, message: str):
        self.chat_history.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.chat_history.append(AIMessage(content=message))

    @traceable(name="Trim Conversation History", run_type="tool")
    def get_trimmed_history(self):
        return trim_messages(
            self.chat_history,
            max_tokens=self.max_tokens,
            strategy=self.strategy,
            token_counter=count_tokens_approximately,
            start_on="human",
            include_system=True,
        )

    @traceable(name="Summarize Conversation", run_type="chain")
    def summarize_if_needed(self):
        if len(self.chat_history) <= self.summary_trigger:
            return

        old_messages = self.chat_history[: -self.keep_last_n]
        recent_messages = self.chat_history[-self.keep_last_n:]

        conversation_text = "\n".join(f"{msg.type.upper()}: {msg.content}" for msg in old_messages)

        summary_prompt = (
            "Summarize the conversation below between a user and an assistant. "
            "Keep key facts, user intent, and important details. Be concise.\n\n"
            f"Existing summary (if any): {self.running_summary}\n\n"
            f"Conversation:\n{conversation_text}"
        )

        summary_response = self.llm.invoke(summary_prompt)
        self.running_summary = summary_response.content

        self.chat_history = [
            SystemMessage(content=f"Conversation summary so far: {self.running_summary}")
        ] + recent_messages

        log_msg(f"Memory compressed: summarized {len(old_messages)} old messages.")

    @traceable(name="Get Memory Context", run_type="tool")
    def get_context_messages(self):
        self.summarize_if_needed()
        return self.get_trimmed_history()

    def clear(self):
        self.chat_history = []
        self.running_summary = ""


class SessionManager:
    def __init__(self, llm, **memory_kwargs):
        self.llm = llm
        self.memory_kwargs = memory_kwargs
        self._sessions: dict[str, ChatMemoryManager] = {}

    def get_memory(self, session_id: str) -> ChatMemoryManager:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMemoryManager(llm=self.llm, **self.memory_kwargs)
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
            return True
        return False
