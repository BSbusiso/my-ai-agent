@'
# My Local AI Assistant

A Tkinter desktop chat app that talks to [Ollama](https://ollama.com), auto-routes each question to the right model, streams replies, times them, and remembers conversation history.

## Models used
- General chat: `qwen2.5:3b`
- Coding (fast): `qwen2.5-coder:3b`
- Coding (deep): `deepseek-coder:6.7b`

## Setup
1. Install [Ollama](https://ollama.com/download) and pull the models:
   ```powershell
   ollama pull qwen2.5:3b
   ollama pull qwen2.5-coder:3b
   ollama pull deepseek-coder:6.7b
