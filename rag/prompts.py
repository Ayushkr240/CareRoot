"""
CareRoot RAG - Prompt Construction

Responsibilities:
- Build a prompt from the latest user question, conversation history,
  and evidence context returned by rag/retrieval.py.
- Instruct the answer model to ground healthcare answers in the supplied
  evidence and be transparent when evidence is insufficient.
- Keep evidence/source information visible to support citation-aware answers.

This module does NOT:
- Retrieve documents or access ChromaDB.
- Search Tavily or scrape websites.
- Invoke Gemini or any other LLM.
"""

from typing import Any, Mapping, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



SYSTEM_PROMPT = """You are CareRoot, a healthcare information assistant.

Answer the user's question clearly and directly using the supplied
retrieved evidence. Keep the response focused on what was asked.

Evidence and accuracy:
1. Base medical claims only on the retrieved evidence.
2. Do not invent medical facts, recommendations, or citations.
3. Cover the relevant details supported by the evidence. If the evidence
   is insufficient or misses an important detail, say so clearly.
4. Distinguish symptoms from causes, risk factors, and complications
   when relevant to the question.
5. Cite relevant source titles and URLs from the evidence. Never invent
   or alter source URLs.
6. Use conversation history only to understand context, not as medical evidence.

Safety:
- Do not diagnose or claim to replace a healthcare professional.
- For urgent or potentially life-threatening symptoms, advise seeking
  urgent local medical care.
- Do not advise starting, stopping, or changing prescribed medication.
- Recommend professional assessment when the question requires
  individualized medical advice.

Retrieved evidence:
<retrieved_evidence>
{context}
</retrieved_evidence>

If the retrieved evidence is empty or insufficient to answer reliably,
state this clearly instead of guessing.
"""



def _normalize_history(
    history: Sequence[Any] | None,
) -> list[BaseMessage]:
    """
    Normalize supported conversation-history formats into LangChain messages.

    Supported items:
    - Existing LangChain BaseMessage instances.
    - Dictionaries such as {"role": "user", "content": "..."} or
      {"role": "assistant", "content": "..."}.
    - Two-item tuples/lists such as ("human", "...") or ("ai", "...").

    Unknown roles and empty messages are skipped instead of guessed.
    """
    normalized: list[BaseMessage] = []

    if not history:
        return normalized

    for item in history:
        if isinstance(item, BaseMessage):
            normalized.append(item)
            continue

        role = None
        content = None

        if isinstance(item, Mapping):
            role = item.get("role") or item.get("type")
            content = item.get("content")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            role, content = item

        if not isinstance(content, str) or not content.strip():
            continue

        normalized_role = str(role or "").lower().strip()

        if normalized_role in {"user", "human"}:
            normalized.append(HumanMessage(content=content))
        elif normalized_role in {"assistant", "ai"}:
            normalized.append(AIMessage(content=content))

    return normalized


def build_prompt(
    question: str,
    context: str,
    conversation_history: Sequence[Any] | None = None,
) -> ChatPromptTemplate:
    """
    Build a LangChain ChatPromptTemplate for CareRoot.

    Args:
        question: The latest user question.
        context: Formatted retrieved evidence from retrieval.py.
        conversation_history: Optional recent conversation history. Accepts
            LangChain messages, role/content dictionaries, or role/content pairs.

    Returns:
        A ChatPromptTemplate containing system instructions, history placeholder,
        and the latest user question.

    Note:
        This returns a template, not an invoked LLM response. The caller should
        format it or pass it to a model in the LangGraph answer-generation node.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")

    if not isinstance(context, str):
        raise TypeError("context must be a string.")

    history_messages = _normalize_history(conversation_history)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="conversation_history"),
            ("human", "Latest healthcare question:\n{question}"),
        ]
    )

    # Return a partial template with context/history bound, leaving the
    # question as the only required variable for the caller.
    return prompt.partial(
        context=context.strip() or "No relevant source documents were retrieved.",
        conversation_history=history_messages,
    )


def format_prompt_messages(
    question: str,
    context: str,
    conversation_history: Sequence[Any] | None = None,
) -> list[BaseMessage]:
    """
    Convenience helper that returns the fully formatted message list.

    Useful for inspecting/testing the prompt without calling an LLM.
    """
    prompt = build_prompt(
        question=question,
        context=context,
        conversation_history=conversation_history,
    )
    return prompt.format_messages(question=question.strip())
