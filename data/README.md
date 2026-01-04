# Data Storage Directory

This directory contains runtime data for the chatbot system.

## Structure

- **memories/** - Database storage for conversation memories
  - `conversations.db` - SQLite database for mid-term memory (summaries)
  - `vector_db/` - ChromaDB vector database for long-term semantic memory

- **logs/** - System and conversation logs
  - Application logs
  - Error logs
  - Conversation logs

- **recordings/** - Optional audio recordings
  - User voice recordings
  - System audio outputs
  - Debug audio files

## Note

These directories are auto-created by the application if they don't exist.
Data files are not tracked in git (see .gitignore).