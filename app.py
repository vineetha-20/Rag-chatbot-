import streamlit as st
from rag import load_file, build_index, retrieve, generate, save_index, load_saved_index
import uuid
from dotenv import load_dotenv
import os
import sqlite3
from datetime import datetime

load_dotenv()

# -----------------------
# Config from .env
# -----------------------
APP_TITLE = os.getenv("APP_TITLE", "RAG Chat Application")
MAX_FILES = int(os.getenv("MAX_FILES", 10))
DB_PATH   = os.getenv("DB_PATH", "chats.db")

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"📄 {APP_TITLE}")

# -----------------------
# SQLite setup
# -----------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)
    conn.commit()
    conn.close()

def create_chat(chat_id, title):
    conn = get_db()
    conn.execute(
        "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
        (chat_id, title, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_all_chats():
    conn = get_db()
    chats = conn.execute(
        "SELECT * FROM chats ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return chats

def get_messages(chat_id):
    conn = get_db()
    messages = conn.execute(
        "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
        (chat_id,)
    ).fetchall()
    conn.close()
    return messages

def save_message(chat_id, role, content):
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def delete_chat(chat_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()

def update_chat_title(chat_id, title):
    conn = get_db()
    conn.execute(
        "UPDATE chats SET title = ? WHERE id = ?",
        (title, chat_id)
    )
    conn.commit()
    conn.close()

# -----------------------
# Initialize DB
# -----------------------
init_db()

# -----------------------
# Session State
# -----------------------
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.docs = None

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = set()

# -----------------------
# Load saved index on startup
# -----------------------
if st.session_state.index is None:
    saved_index, saved_docs = load_saved_index()
    if saved_index is not None:
        st.session_state.index = saved_index
        st.session_state.docs = saved_docs
        st.session_state.loaded_files.add("📦 restored from cache")

# -----------------------
# Sidebar — Chats
# -----------------------
st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    new_id = str(uuid.uuid4())
    create_chat(new_id, f"Chat {datetime.now().strftime('%d %b %H:%M')}")
    st.session_state.current_chat = new_id

# Load all chats from DB
all_chats = get_all_chats()

# Default to first chat if none selected
if st.session_state.current_chat is None and all_chats:
    st.session_state.current_chat = all_chats[0]["id"]

for chat in all_chats:
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        is_active = chat["id"] == st.session_state.current_chat
        label = f"{'▶ ' if is_active else ''}{chat['title']}"
        if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
            st.session_state.current_chat = chat["id"]
    with col2:
        if st.button("🗑", key=f"del_{chat['id']}"):
            delete_chat(chat["id"])
            if st.session_state.current_chat == chat["id"]:
                remaining = [c for c in all_chats if c["id"] != chat["id"]]
                st.session_state.current_chat = remaining[0]["id"] if remaining else None
            st.rerun()

# -----------------------
# Sidebar — File Upload
# -----------------------
st.sidebar.header("📂 Upload Files")
files = st.sidebar.file_uploader(
    "Upload TXT, PDF, or DOCX files",
    type=["txt", "pdf", "docx", "doc"],
    accept_multiple_files=True
)

if files:
    if len(files) > MAX_FILES:
        st.sidebar.warning(f"⚠️ Max {MAX_FILES} files allowed. Only first {MAX_FILES} will be processed.")
        files = files[:MAX_FILES]

    new_files = [f for f in files if f.name not in st.session_state.loaded_files]

    if new_files:
        all_docs = list(st.session_state.docs) if st.session_state.docs else []

        # File reading progress
        file_progress = st.sidebar.progress(0, text="Reading files...")
        for i, file in enumerate(new_files):
            try:
                docs = load_file(file)
                all_docs.extend(docs)
                st.session_state.loaded_files.add(file.name)
            except Exception as e:
                st.sidebar.error(f"❌ {file.name}: {e}")
            file_progress.progress(
                (i + 1) / len(new_files),
                text=f"Reading {file.name}..."
            )
        file_progress.empty()

        if all_docs:
            total_chunks = len(all_docs)
            st.sidebar.markdown(f"📊 Total chunks: **{total_chunks}**")

            # Chunk level embedding progress
            embed_progress = st.sidebar.progress(
                0, text=f"Embedding 0 of {total_chunks} chunks..."
            )

            def update_progress(completed, total):
                embed_progress.progress(
                    completed / total,
                    text=f"Embedding {completed} of {total} chunks..."
                )

            st.session_state.index = build_index(
                all_docs, progress_callback=update_progress
            )
            st.session_state.docs = all_docs

            # Save to disk so next startup is instant
            save_index(st.session_state.index, st.session_state.docs)
            embed_progress.empty()
            st.sidebar.success(
                f"✅ {len(st.session_state.loaded_files)} file(s) loaded — index saved"
            )

# Show loaded files — only once outside the if files block
if st.session_state.loaded_files:
    st.sidebar.markdown("**Loaded files:**")
    for fname in st.session_state.loaded_files:
        st.sidebar.markdown(f"- {fname}")

# -----------------------
# Main Area — Chat
# -----------------------
if st.session_state.current_chat is None:
    st.info("Click '➕ New Chat' in the sidebar to start.")
else:
    messages = get_messages(st.session_state.current_chat)

    if not messages:
        st.info("No messages yet. Upload a file and ask a question!")

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask something...")

    if query:
        if st.session_state.index is None:
            st.warning("Please upload a file first.")
        else:
            # Save user message
            save_message(st.session_state.current_chat, "user", query)

            # Auto title chat from first question
            messages = get_messages(st.session_state.current_chat)
            if len(messages) == 1:
                title = query[:40] + "..." if len(query) > 40 else query
                update_chat_title(st.session_state.current_chat, title)

            # Generate and save answer
            context = retrieve(query, st.session_state.index, st.session_state.docs)
            answer = generate(context, query)
            save_message(st.session_state.current_chat, "assistant", answer)

            st.rerun()  