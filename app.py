import os
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from io import BytesIO
import time
import numpy as np

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

def configure_environment() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        genai.configure(api_key=api_key)


def create_gemini_model() -> Any:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    # Use a reasonably capable model; adjust as needed
    return genai.GenerativeModel("gemini-2.5-pro")

# Difficulty guidance used to steer question generation
DIFFICULTY_GUIDANCE: Dict[str, str] = {
    "Beginner": (
        "Questions should focus on foundational concepts, syntax, and basic definitions. "
        "Avoid complex system design or open-ended scenario-based questions."
    ),
    "Intermediate": (
        "Questions must require the candidate to apply knowledge, combine multiple concepts, and solve common challenges. "
        "Focus on practical problem-solving."
    ),
    "Advanced": (
        "Questions must focus on system design, architecture, performance optimization, and ambiguity resolution. "
        "Questions should be open-ended, requiring the candidate to defend trade-offs and handle edge cases."
    ),
}

def generate_interview_questions(role: str, model: Any, difficulty: str = "Intermediate", num_questions: int = 8) -> List[str]:
    if model is None:
        # Fallback set if no API key configured
        return [
            f"Tell me about yourself and why you want the {role} internship.",
            "Describe a project you're proud of. What was your role?",
            "Explain a challenging problem you solved and how you approached it.",
            "How do you handle feedback and collaborate in a team?",
            "Describe how you prioritize tasks under a tight deadline.",
            "What tools or technologies are you most comfortable with?",
            "How do you stay up-to-date and keep learning?",
            "What are your goals for this internship?",
        ]
    difficulty_note = DIFFICULTY_GUIDANCE.get(difficulty, DIFFICULTY_GUIDANCE["Intermediate"]) if isinstance(difficulty, str) else DIFFICULTY_GUIDANCE["Intermediate"]
    prompt = (
        f"Generate {num_questions} concise, role-appropriate interview questions for a {role} intern. "
        f"Difficulty level: {difficulty}. Guidance: {difficulty_note} "
        "Mix technical and behavioral questions. Return questions as a plain numbered list, no extra commentary."
    )
    response = model.generate_content(prompt)
    text = (response.text or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Clean bullets / numbers
    questions: List[str] = []
    for line in lines:
        # Remove leading numbering/bullets like "1. ", "- ", "* "
        cleaned = line
        if cleaned[:2].isdigit() or cleaned[:2] in {"- ", "* "}:
            # naive strip; ensure we keep content
            cleaned = cleaned.split(" ", 1)[-1]
        cleaned = cleaned.lstrip("0123456789.-) ")
        if cleaned:
            questions.append(cleaned)
    # Ensure we return a reasonable set
    return questions[:num_questions] if questions else [
        f"Why are you interested in the {role} internship?",
        "Tell me about a project where you collaborated with others.",
        "Describe a technical concept you recently learned.",
        "How do you approach problem solving under pressure?",
    ]


def evaluate_responses(role: str, qa: List[Dict[str, str]], model: Any) -> Dict[str, Any]:
    # Build structured prompt per user's specification
    responses_text = "\n".join(
        [
            f"Q{idx}: {item['question']}\nA{idx}: {item.get('answer', '(no answer)')}\n"
            for idx, item in enumerate(qa, start=1)
        ]
    )

    json_schema = '{"overallFeedback": "...", "hiringRecommendation": "...", "scores": [{"dimension": "Technical Knowledge", "score": X, "justification": "..."}, {"dimension": "Behavioral Fit", "score": X, "justification": "..."}, {"dimension": "Communication", "score": X, "justification": "..."}]}'
    prompt = (
        f"You are a senior hiring manager tasked with evaluating a candidate for a {role} intern position. "
        "Evaluate the candidate's performance across the following dimensions, providing a score from 1 to 10 for each and a brief justification: "
        "Technical Knowledge, Behavioral Fit, Communication. Provide an overall hiring recommendation (Yes, No, or Maybe) and a short professional summary. "
        "Return the output as a single JSON object with the following structure using DOUBLE QUOTES for all keys and strings: "
        f"{json_schema}. "
        "Respond with JSON ONLY. Do not include markdown, code fences, or commentary. "
        f"Here are the candidate's questions and responses:\n\n{responses_text}"
    )

    if model is None:
        # Fallback heuristic if no API key
        answered = sum(1 for item in qa if item.get('answer'))
        ratio = max(1, answered) / max(1, len(qa))
        score_base = int(5 + 5 * ratio)
        return {
            "overallFeedback": "Offline evaluation fallback. Provide a Gemini API key for AI-driven insights.",
            "hiringRecommendation": "Maybe",
            "scores": [
                {"dimension": "Technical Knowledge", "score": score_base, "justification": "Baseline offline score."},
                {"dimension": "Behavioral Fit", "score": score_base, "justification": "Shows potential; more detail needed."},
                {"dimension": "Communication", "score": score_base, "justification": "Clear enough; add structure and depth."},
            ],
        }

    response = model.generate_content(prompt)
    text = (response.text or "").strip()

    # Try parsing as JSON; if it fails, attempt lenient extraction/fixes
    import json, re, ast
    def strip_code_fences(s: str) -> str:
        s = s.strip()
        # Remove ```json ... ``` wrappers if present
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        if s.endswith("```"):
            s = re.sub(r"\n```$", "", s)
        return s.strip()

    def extract_json_object(s: str) -> str:
        # Extract the first top-level JSON object conservatively
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            return s[start:end+1]
        return s

    def try_parse(s: str) -> Dict[str, Any]:
        s0 = strip_code_fences(s)
        s1 = extract_json_object(s0)
        try:
            return json.loads(s1)
        except Exception:
            pass
        # Remove trailing commas before } or ]
        s2 = re.sub(r",\s*([}\]])", r"\1", s1)
        try:
            return json.loads(s2)
        except Exception:
            pass
        # Try Python literal eval as a last resort (handles single quotes)
        try:
            return ast.literal_eval(s1)
        except Exception:
            return {}

    data = try_parse(text)

    # Normalize structure to expected keys
    scores = data.get("scores") or []
    # Ensure dimensions exist in some order
    if not scores or not isinstance(scores, list):
        scores = [
            {"dimension": "Technical Knowledge", "score": 7, "justification": "Good foundation."},
            {"dimension": "Behavioral Fit", "score": 7, "justification": "Positive attitude."},
            {"dimension": "Communication", "score": 7, "justification": "Generally clear."},
        ]

    return {
        "overallFeedback": data.get("overallFeedback", text[:800]),
        "hiringRecommendation": data.get("hiringRecommendation", "Maybe"),
        "scores": scores,
    }

def init_session_state() -> None:
    if "candidate_name" not in st.session_state:
        st.session_state.candidate_name = ""
    if "role" not in st.session_state:
        st.session_state.role = "SDE Intern"
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "Intermediate"
    if "questions" not in st.session_state:
        st.session_state.questions: List[str] = []
    if "responses" not in st.session_state:
        st.session_state.responses: List[str] = []
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "review" not in st.session_state:
        st.session_state.review: Dict[str, Any] = {}
    if "last_transcript" not in st.session_state:
        st.session_state.last_transcript = ""


def read_aloud_button(text: str, key: str) -> None:
    # Inject small JS snippet using Web Speech API (speechSynthesis)
    html = f"""
    <button id='{key}' style='padding:6px 10px;'>🔊 Read Aloud</button>
    <script>
      const btn = document.getElementById('{key}');
      if (btn) {{
        btn.onclick = () => {{
          const utter = new SpeechSynthesisUtterance({text!r});
          utter.rate = 1.0;
          utter.pitch = 1.0;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(utter);
        }};
      }}
    </script>
    """
    st.components.v1.html(html, height=40)


def build_pdf(report: Dict[str, Any], candidate_name: str, role: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 1 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, y, "AI Interview Report")
    y -= 0.3 * inch

    c.setFont("Helvetica", 11)
    c.drawString(1 * inch, y, f"Candidate: {candidate_name}")
    y -= 0.2 * inch
    c.drawString(1 * inch, y, f"Role: {role}")
    y -= 0.3 * inch

    def write_section(title: str, body: str, score: Any | None = None):
        nonlocal y
        if y < 1.2 * inch:
            c.showPage()
            y = height - 1 * inch
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1 * inch, y, title)
        y -= 0.22 * inch
        c.setFont("Helvetica", 10)
        if score is not None:
            c.drawString(1 * inch, y, f"Score: {score}")
            y -= 0.2 * inch
        for line in body.split("\n"):
            for chunk in [line[i:i+100] for i in range(0, len(line), 100)]:
                if y < 1.0 * inch:
                    c.showPage()
                    y = height - 1 * inch
                    c.setFont("Helvetica", 10)
                c.drawString(1 * inch, y, chunk)
                y -= 0.18 * inch
        y -= 0.12 * inch

    # New structure: scores is a list of {dimension, score, justification}
    def get_score(dim: str) -> Dict[str, Any]:
        for s in report.get("scores", []) or []:
            if s.get("dimension") == dim:
                return s
        return {"dimension": dim, "score": "-", "justification": ""}

    tech = get_score("Technical Knowledge")
    beh = get_score("Behavioral Fit")
    comm = get_score("Communication")

    write_section("Technical Knowledge", tech.get("justification", ""), tech.get("score"))
    write_section("Behavioral Fit", beh.get("justification", ""), beh.get("score"))
    write_section("Communication", comm.get("justification", ""), comm.get("score"))

    summary = report.get("overallFeedback", "")
    write_section("Summary", summary)

    c.setFont("Helvetica-Bold", 12)
    if y < 1.0 * inch:
        c.showPage()
        y = height - 1 * inch
    recommendation = report.get("hiringRecommendation", "Maybe")
    c.drawString(1 * inch, y, f"Recommendation: {recommendation}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

def main() -> None:
    st.set_page_config(page_title="AI Interviewer", layout="wide")
    configure_environment()
    init_session_state()

    model = create_gemini_model()

    # Sidebar: Candidate setup
    with st.sidebar:
        st.header("Candidate Setup")
        st.session_state.candidate_name = st.text_input("Candidate name", value=st.session_state.candidate_name)
        role_default = st.session_state.role
        st.session_state.role = st.selectbox(
            "Role",
            ["SDE Intern", "Data Science Intern", "Product Intern", "Design Intern", "Custom"],
            index=["SDE Intern", "Data Science Intern", "Product Intern", "Design Intern", "Custom"].index(role_default) if role_default in ["SDE Intern", "Data Science Intern", "Product Intern", "Design Intern", "Custom"] else 0,
        )
        if st.session_state.role == "Custom":
            custom_role = st.text_input("Custom role", value="Intern")
            effective_role = custom_role.strip() or "Intern"
        else:
            effective_role = st.session_state.role

        st.session_state.difficulty = st.selectbox(
            "Difficulty",
            ["Beginner", "Intermediate", "Advanced"],
            index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.difficulty) if st.session_state.get("difficulty") in ["Beginner", "Intermediate", "Advanced"] else 1,
        )

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Generate Questions", use_container_width=True):
                st.session_state.questions = generate_interview_questions(effective_role, model, st.session_state.difficulty)
                st.session_state.responses = [""] * len(st.session_state.questions)
                st.session_state.current_idx = 0
                st.session_state.role = effective_role
                st.session_state.review = {}
        with cols[1]:
            if st.button("Reset", use_container_width=True):
                st.session_state.questions = []
                st.session_state.responses = []
                st.session_state.current_idx = 0
                st.session_state.review = {}

        if not os.getenv("GEMINI_API_KEY"):
            st.info("Add GEMINI_API_KEY to your .env for AI features. Fallback content will be used.")

    st.title("Face-to-Face AI Interviewer (Text-first)")

    if not st.session_state.questions:
        st.write("Use the sidebar to generate interview questions.")
        st.stop()

    # Layout: left Q&A, right navigation
    left, right = st.columns([3, 1])

    with right:
        st.subheader("Questions")
        nav_buttons = []
        for i, q in enumerate(st.session_state.questions):
            answered = bool(st.session_state.responses[i].strip())
            label = f"{i+1} {'✅' if answered else '•'}"
            if st.button(label, key=f"nav-{i}"):
                st.session_state.current_idx = i
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("End Interview"):
            qa = [
                {"question": q, "answer": st.session_state.responses[i]}
                for i, q in enumerate(st.session_state.questions)
            ]
            st.session_state.review = evaluate_responses(st.session_state.role, qa, model)

    with left:
        idx = st.session_state.current_idx
        question = st.session_state.questions[idx]
        st.subheader(f"Question {idx+1}")
        st.write(question)

        read_aloud_button(question, key=f"tts-{idx}")

        st.markdown("#### Webcam & Microphone")
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        })
        webrtc_ctx = webrtc_streamer(
            key="media",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": True},
            async_processing=True,
        )

        def collect_audio_seconds(ctx, seconds: float = 5.0) -> bytes:
            if ctx is None or ctx.audio_receiver is None:
                return b""
            start = time.time()
            samples: list[np.ndarray] = []
            sample_rate = None
            while time.time() - start < seconds:
                frames = ctx.audio_receiver.get_frames(timeout=1)
                for frame in frames:
                    if isinstance(frame, av.AudioFrame):
                        arr = frame.to_ndarray()
                        # arr shape: (channels, samples) -> transpose to (samples, channels)
                        if arr.ndim == 2:
                            arr = arr.T
                        else:
                            arr = np.expand_dims(arr, axis=-1)
                        # Convert to float32 range [-1,1] if int16
                        if arr.dtype == np.int16:
                            data = (arr / 32768.0).astype(np.float32)
                        else:
                            data = arr.astype(np.float32)
                        samples.append(data)
                        if sample_rate is None:
                            sample_rate = frame.sample_rate
                time.sleep(0.01)
            if not samples or sample_rate is None:
                return b""
            audio = np.concatenate(samples, axis=0)
            buf = BytesIO()
            try:
                import soundfile as sf
                sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
                return buf.getvalue()
            finally:
                buf.close()

        def transcribe_wav_bytes(wav_bytes: bytes) -> str:
            if not wav_bytes:
                return ""
            # Deepgram
            dg_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
            if dg_key:
                import requests
                headers = {
                    "Authorization": f"Token {dg_key}",
                    "Content-Type": "audio/wav",
                }
                params = {"smart_format": True}
                r = requests.post("https://api.deepgram.com/v1/listen", headers=headers, params=params, data=wav_bytes, timeout=60)
                if r.ok:
                    j = r.json()
                    try:
                        return j["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
                    except Exception:
                        pass
            # Vosk offline
            vosk_model = os.getenv("VOSK_MODEL_PATH", "").strip()
            if vosk_model:
                try:
                    import vosk, wave, tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tf.write(wav_bytes)
                        temp_path = tf.name
                    wf = wave.open(temp_path, "rb")
                    model = vosk.Model(vosk_model)
                    rec = vosk.KaldiRecognizer(model, wf.getframerate())
                    rec.SetWords(True)
                    import json as _json
                    result_text = []
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            res = _json.loads(rec.Result())
                            if res.get("text"):
                                result_text.append(res["text"])
                    res = _json.loads(rec.FinalResult())
                    if res.get("text"):
                        result_text.append(res["text"])
                    wf.close()
                    return " ".join(result_text).strip()
                except Exception:
                    pass
            return ""

        c_a1, c_a2 = st.columns([1, 1])
        with c_a1:
            if st.button("Transcribe voice to text"):
                wav_bytes = collect_audio_seconds(webrtc_ctx, seconds=5.0)
                text = transcribe_wav_bytes(wav_bytes)
                if text:
                    st.session_state.last_transcript = text
                    st.session_state.responses[idx] = (st.session_state.get(f"resp-{idx}", "") + ("\n" if st.session_state.get(f"resp-{idx}") else "") + text).strip()
                    st.success("Transcribed and appended to your response.")
                else:
                    st.warning("Could not transcribe. Ensure a provider key or VOSK model is set.")
        with c_a2:
            if st.session_state.last_transcript:
                st.info(f"Last transcript: {st.session_state.last_transcript[:200]}")

        st.text_area(
            "Your response",
            key=f"resp-{idx}",
            value=st.session_state.responses[idx],
            height=160,
            on_change=lambda i=idx: st.session_state.responses.__setitem__(i, st.session_state.get(f"resp-{i}", "")),
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Save Response"):
                st.session_state.responses[idx] = st.session_state.get(f"resp-{idx}", "").strip()
                st.success("Response saved.")
        with c2:
            if st.button("Previous", disabled=idx == 0):
                st.session_state.current_idx = max(0, idx - 1)
        with c3:
            if st.button("Next", disabled=idx >= len(st.session_state.questions) - 1):
                st.session_state.current_idx = min(len(st.session_state.questions) - 1, idx + 1)

    # Review & Report
    if st.session_state.review:
        st.markdown("---")
        st.header("AI-Powered Review")
        review = st.session_state.review

        def view_score(node: Dict[str, Any]):
            title = node.get("dimension", "-")
            score = node.get("score", "-")
            justification = node.get("justification", "")
            st.subheader(title)
            cols = st.columns([1, 3])
            with cols[0]:
                st.metric("Score", score)
            with cols[1]:
                st.write(justification)

        for s in review.get("scores", []) or []:
            view_score(s)

        st.subheader("Summary")
        st.write(review.get("overallFeedback", ""))

        st.subheader("Hiring Recommendation")
        st.write(review.get("hiringRecommendation", "Maybe"))

        pdf_bytes = build_pdf(review, st.session_state.candidate_name or "Candidate", st.session_state.role)
        st.download_button(
            label="Download Report (PDF)",
            data=pdf_bytes,
            file_name="interview_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
