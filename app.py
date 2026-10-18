import streamlit as st

from database import (
    create_thread,
    get_threads,
    get_messages,
    save_message,
    update_thread_title,
    delete_thread,
)

from backend_step4 import (
    generate_response,
    generate_chat_title,
)


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="CareRoot",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# Session State
# =========================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.title("🌱 CareRoot")

    st.caption("Evidence-Grounded Healthcare Guide")

    st.divider()

    # ---------------------------------------------
    # New Chat Button
    # ---------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        new_thread_id = create_thread()

        st.session_state.thread_id = new_thread_id

        st.rerun()

    st.divider()

    st.subheader("Conversations")

    # ---------------------------------------------
    # Get conversations
    # ---------------------------------------------

    threads = get_threads()

    if not threads:

        st.caption("No previous conversations.")

    else:

        for thread_id, title, created_at in threads:

            is_current = (
                thread_id == st.session_state.thread_id
            )

            # -----------------------------------------
            # Conversation row
            # -----------------------------------------

            col1, col2 = st.columns([5, 1])

            # -----------------------------------------
            # Chat button
            # -----------------------------------------

            with col1:

                if st.button(
                    f" {title}",
                    key=f"thread_{thread_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):

                    st.session_state.thread_id = thread_id

                    st.rerun()

            # -----------------------------------------
            # Three-dot menu
            # -----------------------------------------

            with col2:

                with st.popover(
                    "⋮",
                    use_container_width=True
                ):
                    if st.button(
                        " Delete",
                        key=f"delete_{thread_id}",
                        use_container_width=True
                    ):
                        # Remember the current sidebar order before deletion.
                        thread_ids = [thread[0] for thread in threads]
                        deleted_index = thread_ids.index(thread_id)

                        # Delete the selected conversation and its messages.
                        delete_thread(thread_id)

                        # If the deleted chat was active, select the chat
                        # immediately below it in the current sidebar order.
                        if st.session_state.thread_id == thread_id:
                            remaining_threads = get_threads()

                            if remaining_threads:
                                next_index = min(deleted_index, len(remaining_threads) - 1)
                                st.session_state.thread_id = remaining_threads[next_index][0]
                            else:
                                st.session_state.thread_id = None

                        st.rerun()

# =========================================================
# Create Initial Chat
# =========================================================

if st.session_state.thread_id is None:
    threads = get_threads()

    if threads:
        st.session_state.thread_id = threads[0][0]
        st.rerun()
    else:
        st.info("No conversations yet. Click '+ New Chat' to start one.")
        st.stop()


# =========================================================
# Main Chat Area
# =========================================================

st.title("CareRoot")

st.caption(
    "Ask questions about health, wellness and your body."
)


# =========================================================
# Load Current Conversation
# =========================================================

messages = get_messages(
    st.session_state.thread_id
)


# =========================================================
# Display Messages
# =========================================================

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input(
    "Ask CareRoot a question..."
)


if user_input:

    # ---------------------------------------------
    # Display User Message
    # ---------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_input)

    # Save user message
    save_message(
        st.session_state.thread_id,
        "user",
        user_input
    )

    # ---------------------------------------------
    # Set title for first message
    # ---------------------------------------------

    if len(messages) == 0:

        title = generate_chat_title(
            user_input,
            thread_id=st.session_state.thread_id,
        )

        update_thread_title(
            st.session_state.thread_id,
            title
        )

    # ---------------------------------------------
    # Get Updated Conversation
    # ---------------------------------------------

    messages = get_messages(
        st.session_state.thread_id
    )

    # ---------------------------------------------
    # Generate Response
    # ---------------------------------------------

    with st.chat_message("assistant"):

        response = generate_response(
            messages,
            thread_id=st.session_state.thread_id,
        )

        st.markdown(response)

    # ---------------------------------------------
    # Save Assistant Response
    # ---------------------------------------------

    save_message(
        st.session_state.thread_id,
        "assistant",
        response
    )

    # Refresh sidebar so the new conversation title
    # and ordering are immediately visible.
    st.rerun()