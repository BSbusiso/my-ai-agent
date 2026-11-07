"""
My Local AI Assistant
---------------------
A Tkinter chat app that talks to Ollama, routes each question to the
right model, streams the reply, times it, and remembers history
across multiple sessions (shown in a sidebar).

Models used:
  - General chat  : qwen2.5:3b
  - Coding (fast) : qwen2.5-coder:3b
  - Coding (deep) : deepseek-coder:6.7b

Commands:
  /new      start a fresh conversation (current one is kept in sidebar)
  /clear    delete ALL saved conversations (with confirmation)
  /chat     force the general model
  /code     force a coding model (complexity-routed)
  /fast     force the light coding model
  /deep     force the strong coding model
  /rotate   force round-robin across coding models
"""

import json
import re
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext

import ollama


# -----------------------------
# Configuration
# -----------------------------

GENERAL_MODEL = "qwen2.5:3b"

CODING_MODELS = [
    "qwen2.5-coder:3b",
    "deepseek-coder:6.7b",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "chat_history.json"
TIMING_LOG = DATA_DIR / "timings.jsonl"

SYSTEM_PROMPT_GENERAL = """
You are my personal AI assistant.
Help me learn Python, AI, machine learning, and software development.
Explain difficult concepts in beginner-friendly language.
Use practical examples and step-by-step instructions.
Be honest when you do not know an answer.
"""

SYSTEM_PROMPT_CODING = """
You are my expert coding assistant.
Focus on writing clean, working code.
Always explain what the code does and why.
Point out bugs, edge cases, and better alternatives.
Prefer Python unless I ask for another language.
Be concise but complete. """

CODING_KEYWORDS = {
    "code", "coding", "program", "programming", "script", "function",
    "class", "method", "variable", "loop", "list", "dict", "dictionary",
    "debug", "error", "bug", "fix", "syntax", "traceback", "exception",
    "python", "javascript", "java", "c++", "sql", "html", "css",
    "algorithm", "compile", "library", "package", "module", "import",
    "api", "json", "regex", "refactor", "unit", "test", "pytest",
}

HARD_SIGNALS = (
    "refactor", "architecture", "design", "optimize", "why does",
    "explain the difference", "step by step", "multiple files",
    "class", "module", "package", "async", "thread", "memory",
    "concurrency", "performance", "scalab",
)


# -----------------------------
# Model routing & selection
# -----------------------------

def detect_category(question: str) -> str:
    """Return 'coding' or 'general'."""
    lowered = question.lower().strip()

    if lowered.startswith("/code") or lowered.startswith("/fast") \
            or lowered.startswith("/deep") or lowered.startswith("/rotate"):
        return "coding"
    if lowered.startswith("/chat") or lowered.startswith("/general"):
        return "general"

    words = set(re.findall(r"[a-z0-9\+#]+", lowered))
    if words & CODING_KEYWORDS:
        return "coding"
    if any(p in lowered for p in ("how do i code", "write a function", "unit test")):
        return "coding"
    return "general"


def pick_coder_model(question: str) -> str:
    """Choose which coding model to use based on the request."""
    lowered = question.lower()

    if lowered.startswith("/fast"):
        return CODING_MODELS[0]
    if lowered.startswith("/deep"):
        return CODING_MODELS[-1]

    word_count = len(question.split())
    is_hard = (
        word_count > 40
        or len(question) > 300
        or "```" in question
        or any(sig in lowered for sig in HARD_SIGNALS)
    )
    return CODING_MODELS[-1] if is_hard else CODING_MODELS[0]


def strip_command_prefix(question: str) -> str:
    """Remove the leading /command (if any)."""
    for prefix in ("/code", "/chat", "/general", "/fast", "/deep", "/rotate"):
        if question.lower().startswith(prefix):
            return question[len(prefix):].strip()
    return question.strip()


class ModelRotator:
    """Cycles through a list of models, one per call."""
    def __init__(self, models):
        self.models = list(models)
        self.index = 0

    def next(self):
        model = self.models[self.index]
        self.index = (self.index + 1) % len(self.models)
        return model


# -----------------------------
# Decorators
# -----------------------------

def timed_stream(func):
    """Time a generator function that yields chunks."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        first_chunk_time = None
        chunk_count = 0
        try:
            for chunk in func(*args, **kwargs):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                chunk_count += 1
                yield chunk
        finally:
            end = time.perf_counter()
            total = end - start
            ttft = (first_chunk_time - start) if first_chunk_time else float("nan")
            print(
                f"[timing] {func.__name__}: "
                f"TTFT={ttft:.2f}s  total={total:.2f}s  chunks={chunk_count}",
                flush=True,
            )
    return wrapper


def with_fallback(models):
    """Try each model in order until one succeeds (pre-first-chunk only)."""
    def decorator(func):
        @wraps(func)
        def wrapper(history, model_name, system_prompt, on_chunk=None, **kwargs):
            ordered = [model_name] + [m for m in models if m != model_name]
            last_error = None

            for candidate in ordered:
                first_chunk_sent = False

                def wrapped_on_chunk(piece):
                    nonlocal first_chunk_sent
                    first_chunk_sent = True
                    if on_chunk:
                        on_chunk(piece)

                try:
                    return func(
                        history, candidate, system_prompt,
                        on_chunk=wrapped_on_chunk, **kwargs,
                    )
                except Exception as error:
                    last_error = error
                    print(f"[fallback] {candidate} failed: {error}")
                    if first_chunk_sent:
                        raise
                    continue

            raise last_error
        return wrapper
    return decorator


# -----------------------------
# Persistent history (sessions)
# -----------------------------

def _migrate(old):
    """Wrap old flat-history format into a session list."""
    if not old:
        return []
    if isinstance(old, list) and old and "role" in old[0]:
        return [{
            "id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "title": "Imported conversation",
            "created": datetime.now().isoformat(timespec="seconds"),
            "messages": old,
        }]
    return old


def load_sessions():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _migrate(data)
    except (json.JSONDecodeError, ValueError):
        return []


def save_sessions(sessions):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DATA_DIR / "chat_history.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)
    tmp.replace(HISTORY_FILE)


def log_timing(model_name, ttft, total, chunks):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "ttft_s": round(ttft, 3) if ttft == ttft else None,
        "total_s": round(total, 3),
        "chunks": chunks,
    }
    with TIMING_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


# -----------------------------
# Ollama call
# -----------------------------

@timed_stream
def stream_reply(model_name, messages):
    """Yield text pieces from Ollama as they arrive."""
    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True,
        keep_alive="30m",
    )
    for chunk in stream:
        piece = chunk.message.content or ""
        if piece:
            yield piece


@with_fallback(CODING_MODELS + [GENERAL_MODEL])
def ask_ai(history, model_name, system_prompt, on_chunk=None, on_timing=None):
    """Send conversation to Ollama, stream the reply, report timing."""
    messages = [
        {"role": "system", "content": system_prompt},
        *history[-20:],
    ]

    start = time.perf_counter()
    ttft = None
    chunks = 0
    parts = []

    for piece in stream_reply(model_name, messages):
        if ttft is None:
            ttft = time.perf_counter() - start
        chunks += 1
        parts.append(piece)
        if on_chunk:
            on_chunk(piece)

    total = time.perf_counter() - start
    if on_timing:
        on_timing(model_name, ttft, total, chunks)

    return "".join(parts)


# -----------------------------
# Tkinter GUI
# -----------------------------

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("My Local AI Assistant")
        self.root.geometry("960x680")
        self.root.minsize(720, 500)

        # --- sessions ---
        self.sessions = load_sessions()
        if self.sessions:
            self.current_session_index = len(self.sessions) - 1
            self.history = list(self.sessions[-1]["messages"])
        else:
            self.sessions = []
            self.current_session_index = None
            self.history = []

        self.coder_pool = ModelRotator(CODING_MODELS)

        self._build_widgets()
        self._check_ollama()

    # ---- Session management ----

    def _refresh_session_list(self):
        """Redraw the sidebar listbox from self.sessions."""
        self.session_listbox.delete(0, tk.END)
        for s in self.sessions:
            title = s["title"]
            if len(title) > 28:
                title = title[:27] + "…"
            self.session_listbox.insert(tk.END, "  " + title)
        if self.current_session_index is not None:
            self.session_listbox.selection_clear(0, tk.END)
            self.session_listbox.selection_set(self.current_session_index)

    def _on_session_click(self, event):
        sel = self.session_listbox.curselection()
        if not sel:
            return
        index = sel[0]
        if index == self.current_session_index:
            return
        self._load_session(index)

    def _load_session(self, index):
        """Switch the active conversation to sessions[index]."""
        # persist current session's messages before switching
        if self.current_session_index is not None:
            self.sessions[self.current_session_index]["messages"] = list(self.history)
            save_sessions(self.sessions)

        self.current_session_index = index
        self.history = list(self.sessions[index]["messages"])

        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self._append(f"Loaded: {self.sessions[index]['title']}\n\n", "system")
        for msg in self.history:
            if msg["role"] == "user":
                self._append(f"You: {msg['content']}\n", "user")
            else:
                self._append("Assistant: ", "assistant")
                self._append(msg["content"] + "\n\n", "assistant")

    # ---- UI construction ----

    def _build_widgets(self):
        # ---- TOP BAR ----
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=(10, 0))

        tk.Label(top, text="Auto-routing: ON",
                 font=("Consolas", 10, "bold"), fg="#0b8043").pack(side=tk.LEFT)
        tk.Button(top, text="New chat", command=self.new_chat).pack(side=tk.RIGHT)
        tk.Button(top, text="Clear saved", command=self.clear_history).pack(
            side=tk.RIGHT, padx=(0, 6))

        # ---- MAIN AREA: sidebar | chat ----
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 5))

        # --- Sidebar ---
        sidebar = tk.Frame(main, width=200, bg="#f4f4f4")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="History", font=("Consolas", 10, "bold"),
                 bg="#f4f4f4", anchor="w", padx=8, pady=6).pack(fill=tk.X)

        lb_frame = tk.Frame(sidebar, bg="#f4f4f4")
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.session_listbox = tk.Listbox(
            lb_frame, font=("Consolas", 9), activestyle="none",
            selectmode=tk.SINGLE, borderwidth=0, highlightthickness=0,
            exportselection=False,
        )
        sb = tk.Scrollbar(lb_frame, command=self.session_listbox.yview)
        self.session_listbox.configure(yscrollcommand=sb.set)
        self.session_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.session_listbox.bind("<<ListboxSelect>>", self._on_session_click)

        # --- Chat display ---
        self.chat_display = scrolledtext.ScrolledText(
            main, wrap=tk.WORD, state=tk.DISABLED,
            font=("Consolas", 11), padx=10, pady=10,
        )
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.chat_display.tag_config(
            "user", foreground="#1a73e8", font=("Consolas", 11, "bold"))
        self.chat_display.tag_config("assistant", foreground="#0b8043")
        self.chat_display.tag_config(
            "system", foreground="#888888", font=("Consolas", 9, "italic"))
        self.chat_display.tag_config(
            "timing", foreground="#b06000", font=("Consolas", 9, "italic"))

        # ---- BOTTOM BAR ----
        bottom = tk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.entry = tk.Entry(bottom, font=("Consolas", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self.entry.bind("<Return>", lambda e: self.send_message())

        tk.Button(bottom, text="Send", width=8,
                  command=self.send_message).pack(side=tk.LEFT, padx=(6, 0))

        self._refresh_session_list()

    # ---- helpers ----

    def _append(self, text, tag):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text, tag)
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _check_ollama(self):
        try:
            models = ollama.list()
            installed = {m.model for m in models.models}
            self._append("Connected to Ollama.\n", "system")
            self._append(
                f"General: {GENERAL_MODEL}\n"
                f"Coding pool: {', '.join(CODING_MODELS)}\n",
                "system",
            )
            missing = [m for m in [GENERAL_MODEL, *CODING_MODELS]
                       if m not in installed and f"{m}:latest" not in installed]
            if missing:
                self._append(
                    f"[warn] not installed: {', '.join(missing)} "
                    f"(run: ollama pull <name>)\n",
                    "system",
                )
            self._append(
                "Commands: /new  /clear  /chat  /code  /fast  /deep  /rotate\n\n",
                "system",
            )
        except Exception as error:
            self._append(f"Could not connect to Ollama: {error}\n", "system")

    # ---- actions ----

    def send_message(self):
        raw = self.entry.get().strip()
        if not raw:
            return
        self.entry.delete(0, tk.END)

        if raw.lower() == "/new":
            self.new_chat()
            return
        if raw.lower() == "/clear":
            self.clear_history()
            return

        category = detect_category(raw)
        question = strip_command_prefix(raw)

        if category == "coding":
            if raw.lower().startswith("/rotate"):
                model_name = self.coder_pool.next()
            else:
                model_name = pick_coder_model(raw)
            system_prompt = SYSTEM_PROMPT_CODING
        else:
            model_name = GENERAL_MODEL
            system_prompt = SYSTEM_PROMPT_GENERAL

        self._append(f"You: {question}\n", "user")
        self._append(f"[{category} -> {model_name}]\n", "system")
        self._append("Assistant: ", "assistant")

        self.history.append({"role": "user", "content": question})

        self.entry.configure(state=tk.DISABLED)

        def on_chunk(piece):
            self._append(piece, "assistant")

        def on_timing(used_model, ttft, total, chunks):
            ttft_str = f"{ttft:.2f}s" if ttft == ttft else "n/a"
            self._append(
                f"\n[timing] first token {ttft_str} | "
                f"total {total:.2f}s | {chunks} chunks | {used_model}\n\n",
                "timing",
            )
            log_timing(used_model, ttft, total, chunks)

        def worker():
            try:
                answer = ask_ai(
                    self.history, model_name, system_prompt,
                    on_chunk=on_chunk, on_timing=on_timing,
                )
                self.history.append({"role": "assistant", "content": answer})

                if self.current_session_index is not None:
                    sess = self.sessions[self.current_session_index]
                    sess["messages"] = list(self.history)
                    if sess["title"] in ("(new conversation)", ""):
                        first_user = next(
                            (m["content"] for m in self.history
                             if m["role"] == "user"),
                            "Conversation",
                        )
                        sess["title"] = first_user[:40]
                    save_sessions(self.sessions)
                    self.root.after(0, self._refresh_session_list)
            except Exception as error:
                self.history.pop()
                self._append(f"\n[error] {error}\n\n", "system")
            finally:
                self.entry.configure(state=tk.NORMAL)
                self.entry.focus_set()

        threading.Thread(target=worker, daemon=True).start()

    def new_chat(self):
        # save current session first
        if self.current_session_index is not None:
            self.sessions[self.current_session_index]["messages"] = list(self.history)
            save_sessions(self.sessions)

        self.sessions.append({
            "id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "title": "(new conversation)",
            "created": datetime.now().isoformat(timespec="seconds"),
            "messages": [],
        })
        self.current_session_index = len(self.sessions) - 1
        self.history = []

        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self._append("Started a fresh conversation.\n\n", "system")
        self._refresh_session_list()

    def clear_history(self):
        if messagebox.askyesno("Clear history",
                            "Delete ALL saved conversations?"):
            self.sessions = []
            self.current_session_index = None
            self.history = []
            save_sessions(self.sessions)
            self.chat_display.configure(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.configure(state=tk.DISABLED)
            self._append("All history cleared.\n\n", "system")
            self._refresh_session_list()


# -----------------------------
# Entry point
# -----------------------------

def main():
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()