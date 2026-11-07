"""
My Local AI Assistant
---------------------
A Tkinter chat app that talks to Ollama, routes each question to the
right model, streams the reply, times it, and remembers history.

Models used:
  - General chat  : qwen2.5:3b
  - Coding (fast) : qwen2.5-coder:3b
  - Coding (deep) : deepseek-coder:6.7b

Commands:
  /new      start a fresh conversation (clears saved history)
  /clear    delete saved history (with confirmation)
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

# Coding models, ordered fast -> strong
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
    """Time a generator function that yields chunks.
    Prints TTFT, total, and chunk count to the terminal."""
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
    """Decorator factory: try each model in order until one succeeds.
    Only retries if the failure happens before the first chunk is sent."""
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
                        # Too late to retry — user already saw partial output.
                        raise
                    continue

            raise last_error
        return wrapper
    return decorator


# -----------------------------
# Persistent history
# -----------------------------

def load_history():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            history = json.load(file)
        if not isinstance(history, list):
            raise ValueError("History must be a list.")
        return history
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Could not read {HISTORY_FILE}. Back it up before repairing it."
        ) from error


def save_history(history):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary_file = DATA_DIR / "chat_history.tmp"
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)
    temporary_file.replace(HISTORY_FILE)


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
        self.root.geometry("860x680")
        self.root.minsize(620, 500)

        try:
            self.history = load_history()
        except RuntimeError as error:
            messagebox.showerror("History error", str(error))
            self.history = []

        self.coder_pool = ModelRotator(CODING_MODELS)

        self._build_widgets()
        self._check_ollama()

    # ---- UI construction ----

    def _build_widgets(self):
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=(10, 0))

        tk.Label(
            top, text="Auto-routing: ON", font=("Consolas", 10, "bold"), fg="#0b8043"
        ).pack(side=tk.LEFT)

        tk.Button(top, text="New chat", command=self.new_chat).pack(side=tk.RIGHT)
        tk.Button(top, text="Clear saved", command=self.clear_history).pack(
            side=tk.RIGHT, padx=(0, 6)
        )

        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state=tk.DISABLED,
            font=("Consolas", 11), padx=10, pady=10,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 5))

        self.chat_display.tag_config(
            "user", foreground="#1a73e8", font=("Consolas", 11, "bold")
        )
        self.chat_display.tag_config("assistant", foreground="#0b8043")
        self.chat_display.tag_config(
            "system", foreground="#888888", font=("Consolas", 9, "italic")
        )
        self.chat_display.tag_config("timing", foreground="#b06000", font=("Consolas", 9, "italic"))

        bottom = tk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.entry = tk.Entry(bottom, font=("Consolas", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self.entry.bind("<Return>", lambda event: self.send_message())

        tk.Button(bottom, text="Send", width=8, command=self.send_message).pack(
            side=tk.LEFT, padx=(6, 0)
        )

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

        # Decide category and model
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

        # Show user message and routing
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
                save_history(self.history)
            except Exception as error:
                self.history.pop()
                self._append(f"\n[error] {error}\n\n", "system")
            finally:
                self.entry.configure(state=tk.NORMAL)
                self.entry.focus_set()

        threading.Thread(target=worker, daemon=True).start()

    def new_chat(self):
        self.history = []
        save_history(self.history)
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self._append("Started a fresh conversation.\n\n", "system")

    def clear_history(self):
        if messagebox.askyesno("Clear history", "Delete saved conversation history?"):
            self.history = []
            save_history(self.history)
            self._append("Conversation history cleared.\n\n", "system")


# -----------------------------
# Entry point
# -----------------------------

def main():
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()