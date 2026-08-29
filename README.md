# NCRTC AI Assistant

A desktop app (`voice_assistant.py`) built with `customtkinter`, styled as an
internship project dashboard for the National Capital Region Transport
Corporation (NCRTC) and its Namo Bharat / RRTS project.

## Layout

- **Header bar** — title, and a live status pill (Ready / Listening... /
  Thinking... / Speaking...) with a colored status dot.
- **Left sidebar** — 5 working navigation buttons:
  - **Home** — welcome overview + quick facts
  - **Stations** — a directory of stations on the Delhi–Ghaziabad–Meerut
    RRTS corridor
  - **Documents** — lists files from a local `documents/` folder (created
    automatically) and lets you open them with one click, or open the
    folder itself in File Explorer/Finder
  - **AI Mode** — the chat interface: scrollable history, text input, Send
    button, and a 🎤 Start/Stop mic button for voice input
  - **About** — project summary

## AI behavior

The assistant (`get_ai_response()`) uses Google's Gemini API (free tier) and
is grounded with real NCRTC/Namo Bharat facts (corridor length, speeds,
station names, operating company, etc.) via a system instruction, and is
told to explain things conversationally, like a knowledgeable human guide,
rather than a generic corporate chatbot. Off-topic questions still get
answered, just more briefly. If no Gemini key is configured, or a call
fails, it falls back to a built-in rule-based offline responder so the app
never breaks.

Everything else — speech-to-text, text-to-speech, and the threading model
so the GUI never freezes — works the same as before.

---

## 1. Requirements

- Python 3.10 or newer
- A working microphone and speakers (optional — text-only mode works fine
  without them)
- Windows, macOS, or Linux

---

## 2. Setup — step by step

### Step 1 — One folder, all 4 files

```
NCRTC-Assistant/
├── voice_assistant.py
├── requirements.txt
├── config.txt
└── README.md
```

A `documents/` subfolder will be created automatically the first time you
run the app and open the Documents page.

### Step 2 — Open a terminal in that folder, install dependencies

```powershell
py -m pip install -r requirements.txt
```

### Step 3 — Add your free Gemini API key

1. Go to https://aistudio.google.com/apikey
2. Sign in with Google → **Create API key** (no credit card required)
3. Open `config.txt` with Notepad
4. Paste your key after the `=`:
   ```
   GEMINI_API_KEY=AIzaSyABCDEF1234567890abcdefgh
   ```
5. Save and close

Skip this and the app still runs, just with the offline fallback responder
instead of real AI answers.

### Step 4 — Run it

```powershell
py voice_assistant.py
```

---

## 3. Using the Documents page

Click **Documents** in the sidebar, then either:
- Click **Open Folder** to reveal the `documents` folder in File Explorer,
  drop your internship files in there (PDFs, Word docs, notes, images,
  anything), then come back and click **Refresh**, or
- Just navigate to `<your project folder>\documents` directly yourself

Each file gets a row with an **Open** button that launches it in your
system's default application for that file type.

---

## 4. Common errors & fixes

### `python`/`pip` not recognized

Use `py` instead of `python`, and `py -m pip` instead of `pip`.

### `PyAudio` fails to install (Windows)

```powershell
py -m pip install pipwin
py -m pipwin install pyaudio
```

### `PyAudio` fails to install (macOS)

```bash
brew install portaudio
pip install pyaudio
```

### `PyAudio` fails to install (Linux)

```bash
sudo apt-get install python3-pyaudio portaudio19-dev
```

### "Microphone not found"

The app detects this and disables voice input gracefully — text chat still
works fully. Check your OS's microphone privacy settings if you expect it
to work.

### Still shows "No Gemini API key detected"

- Confirm you saved `config.txt` after pasting the key
- Confirm `config.txt` is in the same folder as `voice_assistant.py`
- Fully close and re-run the app — the key is only read at startup

### Gemini API errors (quota, invalid key, network)

The app automatically falls back to the offline responder for that message
and prints the reason to the terminal window (not the GUI) with a
`[WARN] Gemini API call failed...` line.

**Rate limits specifically:** Google's free tier has gotten much stricter
since late 2025 — daily quotas for `gemini-2.5-flash` can be as low as
20 requests/day on some projects. This app uses `gemini-2.5-flash-lite`
instead, which has a noticeably higher free daily quota, but you can still
hit it with heavy use. The daily quota resets at **midnight Pacific Time**
(not 24 hours after your last request). Check your live quota at
https://aistudio.google.com under your project's usage/rate-limit page.
If you hit the limit, the app tells you this directly in the chat and
still answers using the offline fallback so the conversation doesn't break.

### Documents won't open

Confirm the file isn't open/locked in another program, and that you have a
default application associated with that file type on your system.

### No sound / TTS not speaking

Linux users may need `espeak`:
```bash
sudo apt-get install espeak
```

---

## 5. Structure

`voice_assistant.py` is organized into:

1. **API key loading** — env var, then `config.txt`
2. **NCRTC knowledge grounding** — the system instruction with factual context
3. **LLM response layer** — `get_ai_response()`
4. **Text-to-speech / speech-to-text engines** — background-threaded
5. **Static content** — the RRTS station list
6. **GUI application** — sidebar navigation + 5 pages, all sharing one
   thread-safe `queue.Queue` for background-thread → GUI updates
