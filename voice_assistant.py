r"""
=======================================================================
 NCRTC AI Assistant  (Voice-Controlled LLM Desktop Assistant)
=======================================================================

SETUP INSTRUCTIONS (also repeated in the README):

1. Install Python 3.10+ (Windows/macOS/Linux).

2. Put all the files from this project (voice_assistant.py,
   requirements.txt, config.txt, README.md) in the SAME folder.

3. Install dependencies:
       py -m pip install -r requirements.txt

4. (Optional but recommended) Get a FREE Gemini API key:
       https://aistudio.google.com/apikey
   Then open "config.txt" in this same folder with Notepad and paste
   your key after the equals sign, e.g.:
       GEMINI_API_KEY=AIzaSyABC123...
   Save the file. No terminal commands needed, and it persists across
   every future run.

   If you skip this, the app still runs fully using a built-in offline
   fallback responder instead of real AI replies.

5. Run the app:
       py voice_assistant.py

=======================================================================
"""

import os
import sys
import queue
import threading
import traceback
import subprocess
from datetime import datetime

import customtkinter as ctk

# ----------------------------------------------------------------------
# Optional / soft-dependency imports.
# The app must still start even if some optional libraries are missing.
# ----------------------------------------------------------------------

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_LIB_AVAILABLE = True
except ImportError:
    GEMINI_LIB_AVAILABLE = False


# =======================================================================
# API KEY LOADING  (env var first, then config.txt next to this script)
# =======================================================================

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_key_from_config_file() -> str:
    config_path = os.path.join(_script_dir(), "config.txt")
    if not os.path.exists(config_path):
        return ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key_name, _, value = line.partition("=")
                    if key_name.strip() == "GEMINI_API_KEY":
                        return value.strip().strip('"').strip("'")
    except Exception as e:
        print(f"[WARN] Could not read config.txt: {e}")
    return ""


def load_gemini_api_key() -> str:
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    return _load_key_from_config_file()


# =======================================================================
# NCRTC KNOWLEDGE GROUNDING
# Real, factual background NCRTC/Namo Bharat context so the assistant
# answers accurately even without live search, and stays on-topic.
# =======================================================================

NCRTC_SYSTEM_INSTRUCTION = """
You are the "NCRTC AI Assistant" — a warm, knowledgeable virtual guide built
for a student internship project about the National Capital Region Transport
Corporation (NCRTC) and its Namo Bharat Regional Rapid Transit System (RRTS).

GROUNDING FACTS (use these; they are accurate as of your training):
- NCRTC (National Capital Region Transport Corporation) is a joint venture
  of the Government of India and the states of Delhi, Haryana, Rajasthan,
  and Uttar Pradesh, mandated to implement the Namo Bharat / RRTS project
  across the National Capital Region.
- The Namo Bharat network's first corridor is Delhi–Ghaziabad–Meerut,
  82 km long, connecting Delhi to western Uttar Pradesh. It is fully
  operational, with trains designed for 180 km/h and operating at up to
  160 km/h, covering Delhi to Meerut in under an hour.
- Two more Phase-1 corridors are planned: Delhi–Panipat–Karnal and
  Delhi–Gurugram–SNB (towards Alwar, Rajasthan). All three converge at
  Sarai Kale Khan station in Delhi for seamless interchange.
- Major stations on the Delhi–Ghaziabad–Meerut corridor include: Sarai
  Kale Khan, New Ashok Nagar, Sahibabad, Ghaziabad, Guldhar, Duhai,
  Duhai Depot, Muradnagar, Modi Nagar South, Modi Nagar North, Meerut
  South, Shatabdi Nagar, Begumpul, and Modipuram.
- The trains are branded "Namo Bharat"; six-car aerodynamic trainsets
  (expandable to eight) manufactured by Alstom in Savli, Gujarat, with
  a premium coach and a reserved coach for women, capacity ~1,750
  passengers. The corridor is operated by DB RRTS Operations India (a
  Deutsche Bahn subsidiary) under the "RapidX" service brand.
- RRTS is distinct from Delhi Metro: RRTS is a semi-high-speed regional
  system connecting cities across NCR with fewer stops and higher
  speeds, while Delhi Metro serves dense intra-city transit.

HOW TO RESPOND:
- Explain things the way a knowledgeable, friendly human engineer or
  guide would — natural, conversational, and clear. Not like a generic
  corporate chatbot, and not overly formal.
- For questions about NCRTC, RRTS, Namo Bharat, stations, engineering
  concepts relevant to metro/rail systems, or the internship context,
  give real depth: specifics, reasoning, and helpful examples.
- If asked something completely unrelated to this domain, still answer
  helpfully and briefly, but you may note this assistant specializes in
  NCRTC/Namo Bharat topics if it feels natural to do so.
- If you're not certain about a very recent or specific figure (fares,
  exact schedules, latest extensions), say so honestly rather than
  guessing, and suggest checking ncrtc.in for the latest official info.
"""


# =======================================================================
# LLM RESPONSE LAYER
# =======================================================================

GEMINI_API_KEY = load_gemini_api_key()
USE_GEMINI = bool(GEMINI_API_KEY) and GEMINI_LIB_AVAILABLE

gemini_model = None
if USE_GEMINI:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            system_instruction=NCRTC_SYSTEM_INSTRUCTION,
        )
    except Exception as e:
        print(f"[WARN] Gemini setup failed: {e}")
        gemini_model = None
        USE_GEMINI = False


def _fallback_dummy_response(prompt: str) -> str:
    """Rule-based offline responder, used when no Gemini key/library works."""
    text = prompt.lower().strip()

    if not text:
        return "I didn't catch that. Could you say it again?"

    greetings = ["hello", "hi", "hey", "yo", "greetings"]
    if any(text.startswith(g) or text == g for g in greetings):
        return (
            "Hello! I'm the NCRTC AI Assistant. Ask me about NCRTC, the "
            "Namo Bharat trains, RRTS stations, or the internship project."
        )

    if "your name" in text:
        return "I'm the NCRTC AI Assistant, built for this internship project."

    if "time" in text:
        return f"The current time is {datetime.now().strftime('%I:%M %p')}."

    if "date" in text:
        return f"Today's date is {datetime.now().strftime('%A, %B %d, %Y')}."

    if "ncrtc" in text:
        return (
            "NCRTC (National Capital Region Transport Corporation) is a "
            "joint venture of the Government of India and the states of "
            "Delhi, Haryana, Rajasthan, and Uttar Pradesh. It runs the "
            "Namo Bharat RRTS project. (This is a short offline answer — "
            "add your Gemini key in config.txt for full detail.)"
        )

    if any(w in text for w in ["thank", "thanks"]):
        return "You're welcome! Let me know if you need anything else."

    if any(w in text for w in ["bye", "exit", "quit", "goodbye"]):
        return "Goodbye! Have a great day."

    return (
        "I'm currently running in offline fallback mode (no Gemini key "
        f"configured), so I can't do deep reasoning, but here's what I "
        f"heard: \"{prompt.strip()}\". Add your key to config.txt and "
        "restart the app for full, detailed AI-powered answers about "
        "NCRTC and Namo Bharat."
    )


def get_ai_response(prompt: str) -> str:
    """
    Main entry point for generating an assistant reply.
    Uses Gemini (with NCRTC grounding) if configured, otherwise falls back
    to a deterministic offline responder.
    """
    if USE_GEMINI and gemini_model is not None:
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.7,
                ),
            )
            reply = response.text
            return reply.strip() if reply else _fallback_dummy_response(prompt)
        except Exception as e:
            err_text = str(e)
            print(f"[WARN] Gemini API call failed, using fallback. Reason: {e}")
            if "429" in err_text or "quota" in err_text.lower():
                return (
                    "I've hit today's free-tier request limit for the Gemini API "
                    "(it resets at midnight Pacific Time). Here's an offline answer "
                    f"instead: {_fallback_dummy_response(prompt)}"
                )
            return _fallback_dummy_response(prompt)
    else:
        return _fallback_dummy_response(prompt)


# =======================================================================
# TEXT-TO-SPEECH LAYER
# =======================================================================

class TextToSpeechEngine:
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.available = TTS_AVAILABLE
        self._queue = queue.Queue()
        self._stop_flag = threading.Event()
        self.engine = None

        if self.available:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 175)
                voices = self.engine.getProperty("voices")
                if voices:
                    self.engine.setProperty("voice", voices[0].id)
            except Exception as e:
                print(f"[WARN] TTS init failed: {e}")
                self.available = False

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while not self._stop_flag.is_set():
            try:
                text = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if text is None:
                continue
            self._speak_now(text)

    def _speak_now(self, text: str):
        if not self.available or self.engine is None:
            return
        try:
            if self.status_callback:
                self.status_callback("Speaking...")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[WARN] TTS speak error: {e}")
        finally:
            if self.status_callback:
                self.status_callback("Ready")

    def speak(self, text: str):
        if not text:
            return
        if self.available:
            self._queue.put(text)

    def shutdown(self):
        self._stop_flag.set()


# =======================================================================
# SPEECH-TO-TEXT LAYER
# =======================================================================

class SpeechToTextEngine:
    def __init__(self, on_text, on_status, on_error):
        self.on_text = on_text
        self.on_status = on_status
        self.on_error = on_error
        self.available = SR_AVAILABLE
        self.recognizer = None
        self.microphone = None
        self._listening = False
        self._thread = None

        if self.available:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                self.microphone = sr.Microphone()
            except Exception as e:
                print(f"[WARN] Microphone init failed: {e}")
                self.available = False

    def is_listening(self):
        return self._listening

    def start(self):
        if not self.available:
            self.on_error(
                "Microphone/speech_recognition not available. "
                "Install `speechrecognition` + `pyaudio` and connect a mic."
            )
            return
        if self._listening:
            return
        self._listening = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._listening = False

    def _listen_loop(self):
        try:
            with self.microphone as source:
                self.on_status("Calibrating for background noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.6)

                while self._listening:
                    self.on_status("Listening...")
                    try:
                        audio = self.recognizer.listen(
                            source, timeout=4, phrase_time_limit=8
                        )
                    except sr.WaitTimeoutError:
                        continue
                    if not self._listening:
                        break

                    self.on_status("Thinking...")
                    try:
                        text = self.recognizer.recognize_google(audio)
                        if text and text.strip():
                            self.on_text(text.strip())
                    except sr.UnknownValueError:
                        self.on_status("Didn't catch that, listening again...")
                    except sr.RequestError as e:
                        self.on_error(f"Speech recognition service error: {e}")
                        break
        except OSError as e:
            self.on_error(f"Microphone not found or unavailable: {e}")
        except Exception as e:
            self.on_error(f"Unexpected voice input error: {e}")
        finally:
            self._listening = False
            self.on_status("Ready")


# =======================================================================
# STATIC CONTENT (Stations / About pages)
# =======================================================================

RRTS_STATIONS = [
    ("Sarai Kale Khan", "Delhi", "Interchange hub for all three Phase-1 corridors"),
    ("New Ashok Nagar", "Delhi", "Connects eastern Delhi residential belts"),
    ("Sahibabad", "Ghaziabad, UP", "First EV charging station on the network"),
    ("Ghaziabad", "Ghaziabad, UP", "Future interchange for Khurja/Hapur extensions"),
    ("Guldhar", "Ghaziabad, UP", "Serves Raj Nagar Extension, Sanjay Nagar"),
    ("Duhai", "Ghaziabad, UP", "Near several educational institutes"),
    ("Duhai Depot", "Ghaziabad, UP", "Main stabling & maintenance depot"),
    ("Muradnagar", "Ghaziabad, UP", "Integrated with Muradnagar bus station"),
    ("Modi Nagar South", "UP", "Elevated 3-level station"),
    ("Modi Nagar North", "UP", "Dual entry/exit with escalators & lifts"),
    ("Meerut South", "Meerut, UP", "Meerut Metro service begins here"),
    ("Shatabdi Nagar", "Meerut, UP", "Serves nearby industrial units"),
    ("Begumpul", "Meerut, UP", "Underground; Meerut's business center hub"),
    ("Modipuram", "Meerut, UP", "Northern terminus; elevated station"),
]


# =======================================================================
# MAIN APPLICATION (GUI)
# =======================================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

BG_MAIN = "#EAEAEC"
BG_SIDEBAR = "#EAEAEC"
BG_CARD = "#FFFFFF"
BG_HEADER = "#FFFFFF"
ACCENT_BLUE = "#4A7FC1"
ACCENT_BLUE_HOVER = "#3D6BA6"
TEXT_DARK = "#1A1A1A"
STATUS_GREEN = "#3CB371"
STATUS_AMBER = "#E0A526"
STATUS_RED = "#D9534F"
CHAT_BUBBLE_USER = "#DCE9FB"
CHAT_BUBBLE_AI = "#F1F1F3"


class VoiceAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NCRTC AI Assistant")
        self.geometry("1000x680")
        self.minsize(760, 520)
        self.configure(fg_color=BG_MAIN)

        self._gui_queue = queue.Queue()
        self._chat_row = 0
        self._current_page = None
        self._nav_buttons = {}

        self._build_layout()

        self.tts = TextToSpeechEngine(status_callback=self._queue_status)
        self.stt = SpeechToTextEngine(
            on_text=self._queue_user_voice_text,
            on_status=self._queue_status,
            on_error=self._queue_error,
        )

        self._show_page("Home")

        if not self.stt.available:
            self._append_chat(
                "System",
                "Voice input unavailable (speech_recognition/pyaudio/mic "
                "not detected). Text chat still works fully.",
            )
        if not self.tts.available:
            self._append_chat(
                "System",
                "Voice output unavailable (pyttsx3 not installed/initialized). "
                "Replies will be text-only.",
            )
        if not USE_GEMINI:
            self._append_chat(
                "System",
                "No Gemini API key detected — running with the built-in "
                "offline fallback responder. Open config.txt in this folder "
                "and paste your free key from https://aistudio.google.com/apikey, "
                "then restart the app.",
            )

        self._poll_gui_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Overall layout: header bar, sidebar, content area
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ---- Header ----
        header = ctk.CTkFrame(self, fg_color=BG_HEADER, height=64, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        icon_lbl = ctk.CTkLabel(
            header, text="🚄", font=ctk.CTkFont(size=26),
        )
        icon_lbl.grid(row=0, column=0, padx=(20, 8), pady=10)

        title_lbl = ctk.CTkLabel(
            header, text="NCRTC AI Assistant",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_DARK,
        )
        title_lbl.grid(row=0, column=1, sticky="w", pady=10)

        status_wrap = ctk.CTkFrame(header, fg_color="transparent")
        status_wrap.grid(row=0, column=2, padx=20)
        self.status_dot = ctk.CTkLabel(
            status_wrap, text="●", font=ctk.CTkFont(size=14),
            text_color=STATUS_GREEN,
        )
        self.status_dot.grid(row=0, column=0, padx=(0, 6))
        self.status_label = ctk.CTkLabel(
            status_wrap, text="Ready", font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_DARK,
        )
        self.status_label.grid(row=0, column=1)

        # ---- Sidebar ----
        sidebar = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, width=220, corner_radius=0)
        sidebar.grid(row=1, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        menu_lbl = ctk.CTkLabel(
            sidebar, text="MENU", font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_DARK,
        )
        menu_lbl.pack(pady=(24, 16))

        nav_items = [
            ("Home", "🏠"),
            ("Stations", "🚉"),
            ("Documents", "📄"),
            ("AI Mode", "🤖"),
            ("About", "ℹ️"),
        ]
        for name, icon in nav_items:
            btn = ctk.CTkButton(
                sidebar, text=f"{icon}  {name}", anchor="w",
                fg_color=ACCENT_BLUE, hover_color=ACCENT_BLUE_HOVER,
                text_color="white", corner_radius=8, height=42,
                font=ctk.CTkFont(size=14, weight="bold"),
                command=lambda n=name: self._show_page(n),
            )
            btn.pack(fill="x", padx=18, pady=6)
            self._nav_buttons[name] = btn

        # ---- Content container (pages stack here) ----
        self.content = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        self.content.grid(row=1, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.pages = {}
        self.pages["Home"] = self._build_home_page()
        self.pages["Stations"] = self._build_stations_page()
        self.pages["Documents"] = self._build_documents_page()
        self.pages["AI Mode"] = self._build_ai_mode_page()
        self.pages["About"] = self._build_about_page()

        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

    def _show_page(self, name: str):
        self._current_page = name
        self.pages[name].tkraise()
        for n, btn in self._nav_buttons.items():
            if n == name:
                btn.configure(fg_color=ACCENT_BLUE_HOVER)
            else:
                btn.configure(fg_color=ACCENT_BLUE)

    # ------------------------------------------------------------------
    # HOME PAGE
    # ------------------------------------------------------------------
    def _build_home_page(self):
        page = ctk.CTkFrame(self.content, fg_color=BG_MAIN)
        page.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            page, text="🏠  Welcome", font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_DARK, anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(24, 12))

        card = ctk.CTkFrame(page, fg_color=BG_CARD, corner_radius=14)
        card.grid(row=1, column=0, sticky="ew", padx=30, pady=(0, 16))
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card, text="🚄 NCRTC AI Assistant",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT_DARK,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 6))

        intro = (
            "This assistant was built as an internship project covering the "
            "National Capital Region Transport Corporation (NCRTC) and its "
            "Namo Bharat Regional Rapid Transit System (RRTS).\n\n"
            "Use the menu on the left to explore station details, project "
            "documents, and to chat with the AI about NCRTC, RRTS, Namo "
            "Bharat, or general engineering questions."
        )
        ctk.CTkLabel(
            card, text=intro, font=ctk.CTkFont(size=13), text_color="#444444",
            anchor="w", justify="left", wraplength=640,
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 18))

        # Quick facts row
        facts_frame = ctk.CTkFrame(page, fg_color="transparent")
        facts_frame.grid(row=2, column=0, sticky="ew", padx=30)
        facts = [
            ("82 km", "Delhi–Ghaziabad–Meerut corridor"),
            ("160 km/h", "Operating speed"),
            ("14+", "Major stations"),
            ("3", "Planned Phase-1 corridors"),
        ]
        for i, (big, small) in enumerate(facts):
            f = ctk.CTkFrame(facts_frame, fg_color=BG_CARD, corner_radius=12)
            f.grid(row=0, column=i, sticky="nsew", padx=8, pady=4)
            facts_frame.grid_columnconfigure(i, weight=1)
            ctk.CTkLabel(
                f, text=big, font=ctk.CTkFont(size=20, weight="bold"),
                text_color=ACCENT_BLUE,
            ).pack(pady=(14, 0))
            ctk.CTkLabel(
                f, text=small, font=ctk.CTkFont(size=11), text_color="#666666",
                wraplength=140,
            ).pack(pady=(2, 14))

        jump_btn = ctk.CTkButton(
            page, text="💬  Start chatting in AI Mode", height=44,
            fg_color=ACCENT_BLUE, hover_color=ACCENT_BLUE_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=lambda: self._show_page("AI Mode"),
        )
        jump_btn.grid(row=3, column=0, sticky="w", padx=30, pady=20)

        return page

    # ------------------------------------------------------------------
    # STATIONS PAGE
    # ------------------------------------------------------------------
    def _build_stations_page(self):
        page = ctk.CTkFrame(self.content, fg_color=BG_MAIN)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        header = ctk.CTkLabel(
            page, text="🚉  RRTS Stations — Delhi–Ghaziabad–Meerut Corridor",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT_DARK,
            anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(24, 12))

        scroll = ctk.CTkScrollableFrame(page, fg_color=BG_MAIN)
        scroll.grid(row=1, column=0, sticky="nsew", padx=30, pady=(0, 20))
        scroll.grid_columnconfigure(0, weight=1)

        for i, (name, loc, note) in enumerate(RRTS_STATIONS):
            card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=10)
            card.grid(row=i, column=0, sticky="ew", padx=4, pady=5)
            card.grid_columnconfigure(1, weight=1)

            num = ctk.CTkLabel(
                card, text=str(i + 1), font=ctk.CTkFont(size=14, weight="bold"),
                text_color="white", fg_color=ACCENT_BLUE, corner_radius=14,
                width=28, height=28,
            )
            num.grid(row=0, column=0, rowspan=2, padx=(14, 12), pady=12)

            ctk.CTkLabel(
                card, text=f"{name}  ·  {loc}",
                font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_DARK,
                anchor="w",
            ).grid(row=0, column=1, sticky="ew", padx=(0, 14), pady=(10, 0))

            ctk.CTkLabel(
                card, text=note, font=ctk.CTkFont(size=12), text_color="#666666",
                anchor="w", wraplength=600,
            ).grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=(0, 10))

        return page

    # ------------------------------------------------------------------
    # DOCUMENTS PAGE  (lists & opens files from a local "documents" folder)
    # ------------------------------------------------------------------
    def _build_documents_page(self):
        page = ctk.CTkFrame(self.content, fg_color=BG_MAIN)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)

        header = ctk.CTkLabel(
            page, text="📄  Documents", font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_DARK, anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(24, 6))

        self.docs_folder = os.path.join(_script_dir(), "documents")
        info = ctk.CTkLabel(
            page,
            text=(
                f"Drop internship-related files (PDFs, Word docs, notes) "
                f"into this folder, then click Refresh:\n{self.docs_folder}"
            ),
            font=ctk.CTkFont(size=12), text_color="#555555",
            anchor="w", justify="left", wraplength=700,
        )
        info.grid(row=1, column=0, sticky="ew", padx=30, pady=(0, 10))

        btn_row = ctk.CTkFrame(page, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="new", padx=30)

        self.docs_list_frame = ctk.CTkScrollableFrame(page, fg_color=BG_MAIN)
        self.docs_list_frame.grid(row=3, column=0, sticky="nsew", padx=30, pady=(10, 20))
        page.grid_rowconfigure(3, weight=1)
        self.docs_list_frame.grid_columnconfigure(0, weight=1)

        refresh_btn = ctk.CTkButton(
            btn_row, text="🔄  Refresh", width=120,
            fg_color=ACCENT_BLUE, hover_color=ACCENT_BLUE_HOVER,
            command=self._refresh_documents,
        )
        refresh_btn.pack(side="left", padx=(0, 10))

        open_folder_btn = ctk.CTkButton(
            btn_row, text="📂  Open Folder", width=140,
            fg_color="#6C757D", hover_color="#565C61",
            command=self._open_documents_folder,
        )
        open_folder_btn.pack(side="left")

        os.makedirs(self.docs_folder, exist_ok=True)
        self.after(200, self._refresh_documents)

        return page

    def _refresh_documents(self):
        for widget in self.docs_list_frame.winfo_children():
            widget.destroy()

        os.makedirs(self.docs_folder, exist_ok=True)
        try:
            files = sorted(os.listdir(self.docs_folder))
        except Exception as e:
            files = []
            print(f"[WARN] Could not list documents folder: {e}")

        files = [f for f in files if os.path.isfile(os.path.join(self.docs_folder, f))]

        if not files:
            ctk.CTkLabel(
                self.docs_list_frame,
                text="No documents yet. Add files to the folder above, then click Refresh.",
                font=ctk.CTkFont(size=13), text_color="#777777",
            ).grid(row=0, column=0, sticky="w", pady=20)
            return

        for i, fname in enumerate(files):
            row = ctk.CTkFrame(self.docs_list_frame, fg_color=BG_CARD, corner_radius=10)
            row.grid(row=i, column=0, sticky="ew", padx=4, pady=5)
            row.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                row, text=f"📄  {fname}", font=ctk.CTkFont(size=13),
                text_color=TEXT_DARK, anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=14, pady=10)

            open_btn = ctk.CTkButton(
                row, text="Open", width=80, height=28,
                fg_color=ACCENT_BLUE, hover_color=ACCENT_BLUE_HOVER,
                command=lambda f=fname: self._open_document(f),
            )
            open_btn.grid(row=0, column=1, padx=10, pady=8)

    def _open_document(self, filename: str):
        full_path = os.path.join(self.docs_folder, filename)
        try:
            if sys.platform.startswith("win"):
                os.startfile(full_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", full_path])
            else:
                subprocess.Popen(["xdg-open", full_path])
        except Exception as e:
            self._append_chat("System", f"Couldn't open {filename}: {e}")
            self._show_page("AI Mode")

    def _open_documents_folder(self):
        try:
            if sys.platform.startswith("win"):
                os.startfile(self.docs_folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self.docs_folder])
            else:
                subprocess.Popen(["xdg-open", self.docs_folder])
        except Exception as e:
            print(f"[WARN] Could not open documents folder: {e}")

    # ------------------------------------------------------------------
    # AI MODE PAGE  (the chat interface)
    # ------------------------------------------------------------------
    def _build_ai_mode_page(self):
        page = ctk.CTkFrame(self.content, fg_color=BG_MAIN)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        header = ctk.CTkLabel(
            page, text="💬  AI Conversation", font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT_DARK, anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(24, 10))

        self.chat_frame = ctk.CTkScrollableFrame(page, fg_color=BG_MAIN)
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=30, pady=(0, 10))
        self.chat_frame.grid_columnconfigure(0, weight=1)

        welcome = ctk.CTkFrame(self.chat_frame, fg_color=CHAT_BUBBLE_AI, corner_radius=12)
        welcome.grid(row=0, column=0, sticky="ew", padx=4, pady=6)
        welcome.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            welcome,
            text=(
                "🤖 Welcome to the NCRTC AI Assistant\n\n"
                "Ask me anything about:\n"
                "• NCRTC\n• RRTS\n• Namo Bharat\n• Stations\n"
                "• Engineering\n• Internship"
            ),
            font=ctk.CTkFont(size=13), text_color=TEXT_DARK,
            justify="left", anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=14)
        self._chat_row = 1

        input_bar = ctk.CTkFrame(page, fg_color="transparent")
        input_bar.grid(row=2, column=0, sticky="ew", padx=30, pady=(0, 24))
        input_bar.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            input_bar, placeholder_text="Type your message here...", height=40,
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self._on_send_clicked())

        self.send_btn = ctk.CTkButton(
            input_bar, text="Send", width=90, height=40,
            fg_color=ACCENT_BLUE, hover_color=ACCENT_BLUE_HOVER,
            command=self._on_send_clicked,
        )
        self.send_btn.grid(row=0, column=1, padx=(0, 8))

        self.mic_btn = ctk.CTkButton(
            input_bar, text="🎤 Start", width=110, height=40,
            fg_color="#2E7D32", hover_color="#1B5E20",
            command=self._on_mic_clicked,
        )
        self.mic_btn.grid(row=0, column=2)

        return page

    def _append_chat(self, sender: str, message: str):
        # Chat lives on the AI Mode page; jump there isn't forced, but the
        # scrollable frame exists regardless of which page is visible.
        bubble_color = {
            "You": CHAT_BUBBLE_USER,
            "Assistant": CHAT_BUBBLE_AI,
            "System": "#FDF3DC",
        }.get(sender, CHAT_BUBBLE_AI)
        text_color = {
            "You": "#1B4C8C",
            "Assistant": TEXT_DARK,
            "System": "#8A6D1D",
        }.get(sender, TEXT_DARK)

        bubble = ctk.CTkFrame(self.chat_frame, fg_color=bubble_color, corner_radius=12)
        bubble.grid(row=self._chat_row, column=0, sticky="ew", padx=4, pady=6)
        bubble.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            bubble, text=f"{sender} · {datetime.now().strftime('%I:%M %p')}",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=text_color,
            anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 0))

        body = ctk.CTkLabel(
            bubble, text=message, font=ctk.CTkFont(size=13), text_color=TEXT_DARK,
            anchor="w", justify="left", wraplength=680,
        )
        body.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        self._chat_row += 1
        self.after(50, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))

    # ------------------------------------------------------------------
    # ABOUT PAGE
    # ------------------------------------------------------------------
    def _build_about_page(self):
        page = ctk.CTkFrame(self.content, fg_color=BG_MAIN)
        page.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            page, text="ℹ️  About this project",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_DARK,
            anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(24, 12))

        card = ctk.CTkFrame(page, fg_color=BG_CARD, corner_radius=14)
        card.grid(row=1, column=0, sticky="ew", padx=30)
        card.grid_columnconfigure(0, weight=1)

        about_text = (
            "NCRTC AI Assistant\n\n"
            "This desktop application was built as an internship project "
            "exploring the National Capital Region Transport Corporation "
            "(NCRTC) and its Namo Bharat Regional Rapid Transit System.\n\n"
            "Features:\n"
            "• Voice and text chat with an AI grounded in NCRTC/RRTS facts\n"
            "• A station directory for the Delhi–Ghaziabad–Meerut corridor\n"
            "• A documents panel for internship-related files\n"
            "• Offline voice input (speech recognition) and voice output "
            "(text-to-speech)\n\n"
            "Built with Python, customtkinter, and Google's Gemini API."
        )
        ctk.CTkLabel(
            card, text=about_text, font=ctk.CTkFont(size=13), text_color="#333333",
            anchor="w", justify="left", wraplength=680,
        ).grid(row=0, column=0, sticky="ew", padx=24, pady=24)

        return page

    # ------------------------------------------------------------------
    # Thread-safe GUI queue plumbing
    # ------------------------------------------------------------------
    def _queue_status(self, text: str):
        self._gui_queue.put(("status", text))

    def _queue_error(self, text: str):
        self._gui_queue.put(("error", text))

    def _queue_user_voice_text(self, text: str):
        self._gui_queue.put(("voice_text", text))

    def _queue_ai_reply(self, text: str):
        self._gui_queue.put(("ai_reply", text))

    def _poll_gui_queue(self):
        try:
            while True:
                kind, payload = self._gui_queue.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                elif kind == "error":
                    self._set_status("Error")
                    self._append_chat("System", payload)
                    self._reset_mic_button()
                elif kind == "voice_text":
                    self._handle_user_message(payload, from_voice=True)
                elif kind == "ai_reply":
                    self._append_chat("Assistant", payload)
                    self.tts.speak(payload)
                    self._set_status("Ready")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_gui_queue)

    def _set_status(self, text: str):
        color = STATUS_GREEN
        low = text.lower()
        if "error" in low:
            color = STATUS_RED
        elif text not in ("Ready",):
            color = STATUS_AMBER
        self.status_label.configure(text=text)
        self.status_dot.configure(text_color=color)

    # ------------------------------------------------------------------
    # Send (text) flow
    # ------------------------------------------------------------------
    def _on_send_clicked(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._handle_user_message(text, from_voice=False)

    def _handle_user_message(self, text: str, from_voice: bool):
        if self._current_page != "AI Mode":
            self._show_page("AI Mode")
        prefix = "🎤 " if from_voice else ""
        self._append_chat("You", f"{prefix}{text}")
        self._set_status("Thinking...")
        threading.Thread(
            target=self._get_response_worker, args=(text,), daemon=True
        ).start()

    def _get_response_worker(self, prompt: str):
        try:
            reply = get_ai_response(prompt)
        except Exception as e:
            reply = f"Sorry, something went wrong generating a response: {e}"
            traceback.print_exc()
        self._queue_ai_reply(reply)

    # ------------------------------------------------------------------
    # Mic button flow
    # ------------------------------------------------------------------
    def _on_mic_clicked(self):
        if not self.stt.available:
            self._append_chat(
                "System",
                "Voice input isn't available on this system. "
                "Check the requirements/setup instructions.",
            )
            return

        if self.stt.is_listening():
            self.stt.stop()
            self._reset_mic_button()
            self._set_status("Ready")
        else:
            self.stt.start()
            self.mic_btn.configure(text="⏹ Stop", fg_color="#C62828", hover_color="#8E0000")
            self._set_status("Listening...")

    def _reset_mic_button(self):
        self.mic_btn.configure(text="🎤 Start", fg_color="#2E7D32", hover_color="#1B5E20")

    # ------------------------------------------------------------------
    def _on_close(self):
        try:
            self.stt.stop()
            self.tts.shutdown()
        except Exception:
            pass
        self.destroy()
        sys.exit(0)


if __name__ == "__main__":
    app = VoiceAssistantApp()
    app.mainloop()
