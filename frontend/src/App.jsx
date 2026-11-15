import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useEffect, useState } from "react";
import "./App.css";

const initialChats = [
  {
    id: 1,
    title: "Understanding vitamin D deficiency",
    messages: [
      {
        role: "user",
        content: "What are the common symptoms of vitamin D deficiency?",
      },
      {
        role: "assistant",
        content:
          "Vitamin D deficiency can be associated with bone pain, muscle weakness, and fatigue. The symptoms can vary, and a blood test is used to assess vitamin D levels.",
      },
    ],
  },
  {
    id: 2,
    title: "Healthy sleep guidelines",
    messages: [
      {
        role: "user",
        content: "How much sleep do adults need?",
      },
      {
        role: "assistant",
        content:
          "Most adults need at least 7 hours of sleep per night. Consistent sleep schedules and good sleep habits can support healthy sleep.",
      },
    ],
  },
  {
    id: 3,
    title: "Understanding blood pressure",
    messages: [
      {
        role: "user",
        content: "What does blood pressure mean?",
      },
      {
        role: "assistant",
        content:
          "Blood pressure measures the force of blood against artery walls. It is recorded using two numbers: systolic and diastolic pressure.",
      },
    ],
  },
];

function Orb() {
  return (
    <div className="orb-container">
      <div className="orb-glow" />
      <div className="orb">
        <div className="orb-highlight" />
      </div>
    </div>
  );
}

function Sidebar({
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onRenameChat,
  onDeleteChat,
}) {
  const [openMenuId, setOpenMenuId] = useState(null);

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <div className="brand">
          <div className="brand-icon">C</div>
          <span>CareRoot</span>
        </div>

        <button className="new-chat-btn" onClick={onNewChat}>
          <span>＋</span>
          New Chat
        </button>

        <div className="sidebar-label">YOUR CONVERSATIONS</div>

        <div className="chat-list">
          {chats.map((chat) => (
            <div className="chat-item-row" key={chat.id}>
              <button
                className={`chat-item ${
                  activeChatId === chat.id ? "active" : ""
                }`}
                onClick={() => {
                  setOpenMenuId(null);
                  onSelectChat(chat.id);
                }}
              >
                <span className="chat-item-icon">◷</span>
                <span className="chat-item-title">{chat.title}</span>
              </button>

              <div className="chat-actions">
                <button
                  className="chat-menu-trigger"
                  title="Conversation options"
                  onClick={() =>
                    setOpenMenuId(
                      openMenuId === chat.id ? null : chat.id
                    )
                  }
                >
                  ⋮
                </button>

                {openMenuId === chat.id && (
                  <div className="chat-dropdown">
                    <button
                      onClick={() => {
                        setOpenMenuId(null);
                        onRenameChat(chat);
                      }}
                    >
                      <span>✎</span> Rename
                    </button>

                    <button
                      className="delete-option"
                      onClick={() => {
                        setOpenMenuId(null);
                        onDeleteChat(chat);
                      }}
                    >
                      <span>⌫</span> Delete
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}

          {chats.length === 0 && (
            <p className="empty-sidebar">No conversations yet.</p>
          )}
        </div>
      </div>

      <div className="sidebar-bottom">
        <div className="sidebar-profile">
          <div className="profile-avatar">C</div>
          <div className="profile-info">
            <span>CareRoot</span>
            <small>Healthcare Assistant</small>
          </div>
          <span className="profile-menu">•••</span>
        </div>
      </div>
    </aside>
  );
}


function Message({ message }) {
  const isAssistant = message.role === "assistant";

  return (
    <div className={`message-row ${isAssistant ? "assistant-row" : "user-row"}`}>
      {isAssistant && <div className="message-avatar">C</div>}

      <div className={`message-content ${isAssistant ? "assistant-message" : "user-message"}`}>
        {isAssistant ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        ) : (
          message.content
        )}
      </div>

      {!isAssistant && <div className="user-avatar">Y</div>}
    </div>
  );
}


function WelcomeScreen({ onSuggestion }) {
  const suggestions = [
    {
      icon: "✧",
      title: "Understand symptoms",
      description: "Learn about common symptoms and their causes.",
      prompt: "What are some common symptoms I should understand?",
    },
    {
      icon: "♡",
      title: "Healthy lifestyle",
      description: "Explore evidence-based health and wellness guidance.",
      prompt: "What are some evidence-based healthy lifestyle habits?",
    },
    {
      icon: "⌘",
      title: "Medical guidelines",
      description: "Explore trusted recommendations from health authorities.",
      prompt: "How can I find trusted medical guidelines?",
    },
  ];

  return (
    <div className="welcome-screen">
      <Orb />

      <div className="welcome-eyebrow">
        <span className="status-dot" />
        YOUR PERSONAL HEALTHCARE COMPANION
      </div>

      <h1>
        Your health,
        <br />
        <span>your understanding.</span>
      </h1>

      <p className="welcome-description">
        Ask questions. Explore trusted healthcare guidelines.
        Make more informed decisions about your well-being.
      </p>

      <div className="suggestion-grid">
        {suggestions.map((item) => (
          <button
            className="suggestion-card"
            key={item.title}
            onClick={() => onSuggestion(item.prompt)}
          >
            <span className="suggestion-icon">{item.icon}</span>
            <strong>{item.title}</strong>
            <span>{item.description}</span>
            <span className="suggestion-arrow">↗</span>
          </button>
        ))}
      </div>

      <div className="welcome-note">
        Grounded in trusted healthcare sources
      </div>
    </div>
  );
}


const API_BASE = "http://127.0.0.1:8000/api";

function App() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeModal, setActiveModal] = useState(null);
  const [modalTitle, setModalTitle] = useState("");
  const [modalLoading, setModalLoading] = useState(false);

  // Load conversations from FastAPI when React starts.
  useEffect(() => {
    async function loadChats() {
      try {
        const response = await fetch(`${API_BASE}/chats`);

        if (!response.ok) {
          throw new Error("Could not load conversations.");
        }

        const data = await response.json();

        setChats(
          data.chats.map((chat) => ({
            id: chat.thread_id,
            title: chat.title,
            messages: [],
          }))
        );
      } catch (err) {
        setError(err.message);
      }
    }

    loadChats();
  }, []);

  // Load a selected conversation and its messages.
  async function handleSelectChat(threadId) {
    setError("");
    setActiveChatId(threadId);

    try {
      const response = await fetch(`${API_BASE}/chats/${threadId}`);

      if (!response.ok) {
        throw new Error("Could not load this conversation.");
      }

      const data = await response.json();

      setChats((previous) =>
        previous.map((chat) =>
          chat.id === threadId
            ? {
                ...chat,
                title: data.chat.title,
                messages: data.messages,
              }
            : chat
        )
      );
    } catch (err) {
      setError(err.message);
    }
  }

  // Start a fresh conversation.
  function handleNewChat() {
    setActiveChatId(null);
    setInput("");
    setError("");
  }


 
  function handleRenameChat(chat) {
    setModalTitle(chat.title);
    setActiveModal({
      type: "rename",
      chat,
    });
  }

  function handleDeleteChat(chat) {
    setActiveModal({
      type: "delete",
      chat,
    });
  }

  async function confirmRename() {
    const chat = activeModal?.chat;
    const newTitle = modalTitle.trim();

    if (!chat || !newTitle) return;

    setModalLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/chats/${chat.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ title: newTitle }),
      });

      if (!response.ok) {
        throw new Error("Could not rename this conversation.");
      }

      setChats((previous) =>
        previous.map((item) =>
          item.id === chat.id
            ? { ...item, title: newTitle }
            : item
        )
      );

      setActiveModal(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setModalLoading(false);
    }
  }

  async function confirmDelete() {
    const chat = activeModal?.chat;

    if (!chat) return;

    setModalLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/chats/${chat.id}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Could not delete this conversation.");
      }

      setChats((previous) =>
        previous.filter((item) => item.id !== chat.id)
      );

      if (activeChatId === chat.id) {
        setActiveChatId(null);
        setInput("");
      }

      setActiveModal(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setModalLoading(false);
    }
  }



  // Send a message through FastAPI.
  async function handleSendMessage(text = input) {
    const messageText = text.trim();

    if (!messageText || loading) return;

    setLoading(true);
    setError("");

    try {
      let threadId = activeChatId;

      // Create a database conversation if this is a new chat.
      if (!threadId) {
        const createResponse = await fetch(`${API_BASE}/chats`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            title: "New Chat",
          }),
        });

        if (!createResponse.ok) {
          throw new Error("Could not create a new conversation.");
        }

        const createdChat = await createResponse.json();

        threadId = createdChat.thread_id;
        setActiveChatId(threadId);
      }

      // Display the user message immediately.
      setChats((previous) => {
        const existingChat = previous.find(
          (chat) => chat.id === threadId
        );

        if (existingChat) {
          return previous.map((chat) =>
            chat.id === threadId
              ? {
                  ...chat,
                  messages: [
                    ...chat.messages,
                    { role: "user", content: messageText },
                  ],
                }
              : chat
          );
        }

        return [
          {
            id: threadId,
            title: messageText.slice(0, 35),
            messages: [
              { role: "user", content: messageText },
            ],
          },
          ...previous,
        ];
      });

      setInput("");

      // Send the actual message to FastAPI.
      const messageResponse = await fetch(
        `${API_BASE}/chats/${threadId}/messages`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            content: messageText,
          }),
        }
      );

      const result = await messageResponse.json();

      if (!messageResponse.ok) {
        throw new Error(
          result.detail || "Something went wrong while generating a response."
        );
      }

      // Display the actual AI response returned by FastAPI.
      setChats((previous) =>
        previous.map((chat) =>
          chat.id === threadId
            ? {
                ...chat,
                title: result.title || chat.title,
                messages: [
                  ...chat.messages,
                  {
                    role: "assistant",
                    content: result.assistant_message,
                  },
                ],
              }
            : chat
        )
      );

      // Refresh conversation titles from PostgreSQL.
      const chatsResponse = await fetch(`${API_BASE}/chats`);

      if (chatsResponse.ok) {
        const chatsData = await chatsResponse.json();

        setChats((previous) =>
          previous.map((chat) => {
            const updated = chatsData.chats.find(
              (item) => item.thread_id === chat.id
            );

            return updated
              ? { ...chat, title: updated.title }
              : chat;
          })
        );
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const activeChat = chats.find(
    (chat) => chat.id === activeChatId
  );

  return (
    <div className="app-container">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onRenameChat={handleRenameChat}
        onDeleteChat={handleDeleteChat}
      />

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-status">
            <span className="status-dot" />
            Healthcare guidelines assistant
          </div>

          <div className="topbar-right">
            <span className="secure-label">✧ Evidence focused</span>
          </div>
        </header>

        <section className="chat-area">
          {!activeChat || activeChat.messages.length === 0 ? (
            <WelcomeScreen onSuggestion={handleSendMessage} />
          ) : (
            <div className="conversation">
              <div className="conversation-heading">
                <span>CONVERSATION</span>
                <h2>{activeChat.title}</h2>
              </div>

              <div className="messages-list">
                {activeChat.messages.map((message, index) => (
                  <Message key={index} message={message} />
                ))}

                {loading && (
                  <div className="message-row assistant-row">
                    <div className="message-avatar">C</div>
                    <div className="message-content assistant-message">
                      CareRoot is thinking...
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <footer className="chat-footer">
          <form
            className="chat-input-container"
            onSubmit={(event) => {
              event.preventDefault();
              handleSendMessage();
            }}
          >
            <input
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask CareRoot anything about health..."
              aria-label="Message CareRoot"
              disabled={loading}
            />

            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || loading}
              aria-label="Send message"
            >
              {loading ? "..." : "↑"}
            </button>
          </form>

          <p className="disclaimer">
            CareRoot provides informational guidance, not a substitute for
            professional medical advice.
          </p>
        </footer>

      </main>

      {/* CareRoot Rename / Delete Modal */}
      {activeModal && (
        <div
          className="modal-overlay"
          onClick={() => {
            if (!modalLoading) setActiveModal(null);
          }}
        >
          <div
            className="care-root-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            {activeModal.type === "rename" ? (
              <>
                <div className="modal-header">
                  <div>
                    <span className="modal-eyebrow">CONVERSATION SETTINGS</span>
                    <h2 id="modal-title">Rename conversation</h2>
                  </div>

                  <button
                    className="modal-close"
                    onClick={() => setActiveModal(null)}
                    disabled={modalLoading}
                    aria-label="Close dialog"
                  >
                    ×
                  </button>
                </div>

                <p className="modal-description">
                  Give this conversation a new name.
                </p>

                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    confirmRename();
                  }}
                >
                  <input
                    className="modal-input"
                    type="text"
                    value={modalTitle}
                    onChange={(event) => setModalTitle(event.target.value)}
                    placeholder="Enter conversation title"
                    maxLength={100}
                    autoFocus
                    disabled={modalLoading}
                  />

                  <div className="modal-actions">
                    <button
                      type="button"
                      className="modal-button cancel-button"
                      onClick={() => setActiveModal(null)}
                      disabled={modalLoading}
                    >
                      Cancel
                    </button>

                    <button
                      type="submit"
                      className="modal-button save-button"
                      disabled={!modalTitle.trim() || modalLoading}
                    >
                      {modalLoading ? "Saving..." : "Save changes"}
                    </button>
                  </div>
                </form>
              </>
            ) : (
              <>
                <div className="modal-header">
                  <div>
                    <span className="modal-eyebrow">CONVERSATION SETTINGS</span>
                    <h2 id="modal-title">Delete conversation?</h2>
                  </div>

                  <button
                    className="modal-close"
                    onClick={() => setActiveModal(null)}
                    disabled={modalLoading}
                    aria-label="Close dialog"
                  >
                    ×
                  </button>
                </div>

                <div className="delete-chat-details">
                  <div className="delete-icon">
                    <span>⌫</span>
                  </div>

                  <div className="delete-chat-info">
                    <strong>{activeModal.chat.title}</strong>
                    <p>
                      This conversation and its messages will be permanently
                      deleted.
                    </p>
                  </div>
                </div>

                <div className="modal-actions">
                  <button
                    className="modal-button cancel-button"
                    onClick={() => setActiveModal(null)}
                    disabled={modalLoading}
                  >
                    Cancel
                  </button>

                  <button
                    className="modal-button delete-button"
                    onClick={confirmDelete}
                    disabled={modalLoading}
                  >
                    {modalLoading ? "Deleting..." : "Delete chat"}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}


export default App;
