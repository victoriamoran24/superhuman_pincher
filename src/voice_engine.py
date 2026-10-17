import io
import os
import time
import wave
import threading
import numpy as np
import sounddevice as sd
import pyperclip
from pynput.keyboard import Controller, Key
from kokoro_onnx import Kokoro

# Optional STT imports (Groq cloud API vs local faster-whisper)
USE_LOCAL_WHISPER = False
if USE_LOCAL_WHISPER:
    from faster_whisper import WhisperModel
else:
    from groq import Groq


class VoiceEngine:
    def __init__(
        self,
        model_path: str = "kokoro-v1.0.onnx",
        voices_path: str = "voices-v1.0.bin",
        superhuman_callback=None,
    ):
        """
        superhuman_callback: Optional function from Heba's module `fn(raw_text) -> polished_text`
                             that sends brain-dump transcripts through Superhuman Go / Grammarly.
        """
        print("[VoiceEngine] Loading Kokoro TTS...")
        self.kokoro = Kokoro(model_path, voices_path)
        self.keyboard = Controller()
        self.superhuman_callback = superhuman_callback

        # State flags & thread locks
        self.is_recording = False
        self.is_speaking = False
        self.audio_frames = []
        self.mic_stream = None
        self.sample_rate = 16000
        self.last_trigger_time = 0.0
        self.cooldown_seconds = 1.0  # Prevents MediaPipe frame flicker from double-triggering

        # Initialize STT backend
        if USE_LOCAL_WHISPER:
            print("[VoiceEngine] Loading local faster-whisper (base.en)...")
            self.whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
        else:
            # Expects GROQ_API_KEY in environment variables
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

        print("[VoiceEngine] Ready!")

    # ---------------------------------------------------------
    # INTERNAL HELPERS: Cooldown & Audio Earcons (Chimes)
    # ---------------------------------------------------------
    def _can_trigger(self) -> bool:
        now = time.time()
        if now - self.last_trigger_time < self.cooldown_seconds:
            return False
        self.last_trigger_time = now
        return True

    def _play_chime(self, freqs=(523.25, 659.25), duration=0.08):
        """Plays a quick synthesized two-tone beep for instant gesture feedback."""
        def _beep():
            sr = 24000
            tones = []
            for f in freqs:
                t = np.linspace(0, duration, int(sr * duration), False)
                envelope = np.exp(-t * 20)  # Smooth fade-out
                tones.append(0.25 * np.sin(2 * np.pi * f * t) * envelope)
            audio = np.concatenate(tones).astype(np.float32)
            sd.play(audio, sr)
            sd.wait()

        threading.Thread(target=_beep, daemon=True).start()

    def _copy_selected_text(self) -> str:
        """Uses Pynput to copy currently highlighted text into the clipboard."""
        modifier = Key.cmd if os.uname().sysname == "Darwin" else Key.ctrl
        with self.keyboard.pressed(modifier):
            self.keyboard.press("c")
            self.keyboard.release("c")
        time.sleep(0.12)  # Wait briefly for OS clipboard to update
        return pyperclip.paste().strip()

    def _paste_text(self, text: str):
        """Copies text to clipboard and pastes into the active cursor position via Pynput."""
        pyperclip.copy(text)
        time.sleep(0.05)
        modifier = Key.cmd if os.uname().sysname == "Darwin" else Key.ctrl
        with self.keyboard.pressed(modifier):
            self.keyboard.press("v")
            self.keyboard.release("v")

    # ---------------------------------------------------------
    # 1. VOICE OUTPUT: Reading Guide / Explainer Bot (Kokoro)
    # ---------------------------------------------------------
    def speak_text(self, text: str, voice: str = "af_sarah", lang: str = "en-us", speed: float = 1.0):
        """
        Speaks arbitrary text non-blockingly using Kokoro TTS.
         voices: 'af_sarah', 'af_bella', 'am_adam', 'bf_emma'
         langs:  'en-us', 'en-gb', 'es', 'fr-fr', 'ja', 'cmn'
        """
        if not text.strip():
            return

        def _tts_worker():
            try:
                sd.stop()  # Stop any existing audio before starting new speech
                self.is_speaking = True
                samples, sr = self.kokoro.create(text, voice=voice, speed=speed, lang=lang)
                if self.is_speaking:
                    sd.play(samples, sr)
                    sd.wait()
            except Exception as e:
                print(f"[VoiceEngine] TTS Error: {e}")
            finally:
                self.is_speaking = False

        threading.Thread(target=_tts_worker, daemon=True).start()

    def read_highlighted_selection(self, explain_first: bool = False):
        """
        Triggered after Pinch-and-Drag selection or Reading Guide gesture.
        Grabs highlighted text via Pynput Ctrl+C/Cmd+C and reads it aloud (or explains it first).
        """
        if not self._can_trigger():
            return

        selected_text = self._copy_selected_text()
        if not selected_text:
            print("[VoiceEngine] No highlighted text found in clipboard.")
            return

        print(f"[VoiceEngine] Captured selection: {selected_text!r}")
        if explain_first and self.superhuman_callback:
            self._play_chime((587.33, 880.0))  # Thinking chime
            text_to_speak = self.superhuman_callback(selected_text, mode="explain")
        else:
            text_to_speak = selected_text

        self.speak_text(text_to_speak)

    # ---------------------------------------------------------
    # 2. VOICE INPUT: Fist Gesture Mic Toggle & Brain Dump Mode
    # ---------------------------------------------------------
    def toggle_mic(self, mode: str = "brain_dump"):
        """
        Triggered by Fist Gesture.
        1st Fist: Starts recording microphone audio.
        2nd Fist: Stops recording, transcribes via Whisper, routes through Superhuman Go
                  if in 'brain_dump' mode, and pastes into active text field via Pynput.
        """
        if not self._can_trigger():
            return

        if not self.is_recording:
            # --- START RECORDING ---
            sd.stop()
            self.is_speaking = False
            self.audio_frames = []
            self.is_recording = True
            self._play_chime((440.0, 659.25))  # Rising chime = Mic ON
            print(f"[VoiceEngine] Mic ON ({mode} mode) - Speak now...")

            def _mic_callback(indata, frames, time_info, status):
                if self.is_recording:
                    self.audio_frames.append(indata.copy())

            self.mic_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=_mic_callback,
            )
            self.mic_stream.start()
        else:
            # --- STOP RECORDING & PROCESS ---
            self.is_recording = False
            if self.mic_stream:
                self.mic_stream.stop()
                self.mic_stream.close()
                self.mic_stream = None

            self._play_chime((659.25, 440.0))  # Falling chime = Mic OFF
            print("[VoiceEngine] Mic OFF - Transcribing...")
            threading.Thread(
                target=self._process_recording, args=(mode,), daemon=True
            ).start()

    def _process_recording(self, mode: str):
        if not self.audio_frames:
            return

        # Encode recorded frames into an in-memory WAV file
        audio_data = np.concatenate(self.audio_frames, axis=0)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        wav_buffer.seek(0)
        wav_buffer.name = "brain_dump.wav"

        # Transcribe speech to text
        try:
            if USE_LOCAL_WHISPER:
                segments, _ = self.whisper.transcribe(wav_buffer, beam_size=1)
                transcript = " ".join(seg.text for seg in segments).strip()
            else:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=wav_buffer,
                    model="whisper-large-v3-turbo",
                    response_format="text",
                    language="en",
                )
                transcript = transcription.strip()
        except Exception as e:
            print(f"[VoiceEngine] STT Error: {e}")
            return

        if not transcript:
            print("[VoiceEngine] No speech detected.")
            return

        print(f"[VoiceEngine] Raw Transcript: {transcript}")

        # Route through Heba's Superhuman Go / Brain Dump structuring bot if configured
        if mode == "brain_dump" and self.superhuman_callback:
            final_text = self.superhuman_callback(transcript, mode="brain_dump")
        else:
            final_text = transcript

        print(f"[VoiceEngine] Pasting Output: {final_text}")
        self._paste_text(final_text)

    # ---------------------------------------------------------
    # 3. INTERRUPT: Slow Down / Open-Palm Pause Gesture
    # ---------------------------------------------------------
    def pause_all(self):
        """
        Triggered by Open-Palm Pause Gesture.
        Immediately stops Kokoro TTS playback and cancels any active mic recording.
        """
        if not self._can_trigger():
            return

        print("[VoiceEngine] Open-Palm Pause triggered! Halting audio & mic.")
        self.is_speaking = False
        sd.stop()

        if self.is_recording:
            self.is_recording = False
            if self.mic_stream:
                self.mic_stream.stop()
                self.mic_stream.close()
                self.mic_stream = None
            self.audio_frames = []

        self._play_chime((330.0, 261.63), duration=0.06)  # Soft low double-tap


# =========================================================
# STANDALONE KEYBOARD TEST HARNESS (Run `python voice_engine.py`)
# =========================================================
if __name__ == "__main__":
    # Dummy callback simulating Heba's Superhuman Go bot until she connects hers
    def mock_superhuman_bot(text: str, mode: str = "brain_dump") -> str:
        if mode == "explain":
            return f"In simple terms: {text}"
        return f"[Structured Brain Dump]: {text}"

    engine = VoiceEngine(superhuman_callback=mock_superhuman_bot)

    print("\n--- VOICE ENGINE TEST CONTROLS ---")
    print("  [1] + Enter : Test Reading Guide / Speak Highlighted Text")
    print("  [2] + Enter : Toggle Fist Mic (Start/Stop Brain Dump)")
    print("  [3] + Enter : Trigger Open-Palm Pause (Stop TTS / Cancel Mic)")
    print("  [q] + Enter : Quit\n")

    while True:
        cmd = input("Command (1/2/3/q): ").strip().lower()
        if cmd == "1":
            engine.speak_text(
                "Hello team! Kokoro text to speech is running locally and ready for gestures."
            )
        elif cmd == "2":
            engine.toggle_mic(mode="brain_dump")
        elif cmd == "3":
            engine.pause_all()
        elif cmd == "q":
            break