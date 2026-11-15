
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import (
    init_db,
    create_thread,
    get_threads,
    update_thread_title,
    save_message,
    get_messages,
    delete_thread,
)

from langchain_core.messages import HumanMessage, AIMessage


# =========================================================
# Application Startup
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="CareRoot API",
    description="API for the CareRoot healthcare guidelines assistant.",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# CORS Configuration
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Request Models
# =========================================================

class CreateChatRequest(BaseModel):
    title: str = Field(default="New Chat", min_length=1, max_length=100)


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)


class RenameChatRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)


# =========================================================
# Helper Functions
# =========================================================

def find_thread(thread_id: str):
    """Check whether a conversation exists."""

    threads = get_threads()

    for thread in threads:
        if thread[0] == thread_id:
            return {
                "thread_id": thread[0],
                "title": thread[1],
                "created_at": thread[2],
            }

    return None


def convert_to_langchain_messages(messages):
    """Convert stored conversation history into LangChain messages."""

    converted_messages = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            converted_messages.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":
            converted_messages.append(
                AIMessage(content=content)
            )

    return converted_messages


# =========================================================
# Health Check
# =========================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "application": "CareRoot API",
    }


# =========================================================
# Create New Chat
# =========================================================

@app.post("/api/chats")
def create_new_chat(request: CreateChatRequest):

    thread_id = create_thread(request.title)

    return {
        "success": True,
        "thread_id": thread_id,
        "title": request.title,
    }


# =========================================================
# Get All Chats
# =========================================================

@app.get("/api/chats")
def get_all_chats():

    threads = get_threads()

    return {
        "success": True,
        "chats": [
            {
                "thread_id": thread[0],
                "title": thread[1],
                "created_at": thread[2],
            }
            for thread in threads
        ],
    }


# =========================================================
# Get Single Chat With Messages
# =========================================================

@app.get("/api/chats/{thread_id}")
def get_single_chat(thread_id: str):

    thread = find_thread(thread_id)

    if not thread:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    messages = get_messages(thread_id)

    return {
        "success": True,
        "chat": thread,
        "messages": messages,
    }


# =========================================================
# Send Message
# =========================================================

@app.post("/api/chats/{thread_id}/messages")
def send_message(thread_id: str, request: SendMessageRequest):

    thread = find_thread(thread_id)

    if not thread:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    user_message = request.content.strip()

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # Import the existing LangGraph functions.
    # This keeps the existing AI workflow intact.
    from backend_step4 import (
        generate_response,
        generate_chat_title,
    )

    # Retrieve previous conversation history.
    previous_messages = get_messages(thread_id)

    # Save the new user message.
    save_message(
        thread_id,
        "user",
        user_message,
    )

    # Convert history into LangChain message objects.
    conversation_messages = convert_to_langchain_messages(
        previous_messages
    )

    conversation_messages.append(
        HumanMessage(content=user_message)
    )

    # Generate the AI response through the existing LangGraph workflow.
    try:
        assistant_response = generate_response(
            conversation_messages,
            thread_id=thread_id,
        )

        if not assistant_response.strip():
            raise RuntimeError("The AI returned an empty response.")

    except Exception as exc:
        print(
            f"CareRoot response generation failed: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=502,
            detail="CareRoot could not generate a response. Please try again.",
        )

    # Save the assistant response.
    save_message(
        thread_id,
        "assistant",
        assistant_response,
    )

    # Generate a title for the first user message.
    if not previous_messages:
        try:
            title = generate_chat_title(
                user_message,
                thread_id=thread_id,
            )

            if title:
                update_thread_title(thread_id, title)

        except Exception as exc:
            print(
                f"CareRoot title generation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "success": True,
        "thread_id": thread_id,
        "user_message": user_message,
        "assistant_message": assistant_response,
    }


# =========================================================
# Rename Chat
# =========================================================

@app.patch("/api/chats/{thread_id}")
def rename_chat(thread_id: str, request: RenameChatRequest):

    thread = find_thread(thread_id)

    if not thread:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    title = request.title.strip()

    if not title:
        raise HTTPException(
            status_code=400,
            detail="Title cannot be empty.",
        )

    update_thread_title(thread_id, title)

    return {
        "success": True,
        "thread_id": thread_id,
        "title": title,
    }


# =========================================================
# Delete Chat
# =========================================================

@app.delete("/api/chats/{thread_id}")
def remove_chat(thread_id: str):

    thread = find_thread(thread_id)

    if not thread:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    delete_thread(thread_id)

    return {
        "success": True,
        "message": "Conversation deleted successfully.",
        "thread_id": thread_id,
    }
