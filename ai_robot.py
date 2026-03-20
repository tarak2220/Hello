import pyttsx3
import speech_recognition as sr
import sounddevice as sd
import numpy as np
from openai import OpenAI

# -----------------------------
# Text-to-Speech
# -----------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text):
    print("AI:", text)
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# Speech Recognition
# -----------------------------
recognizer = sr.Recognizer()

# -----------------------------
# OpenAI Client (API key from env)
# -----------------------------
client = OpenAI()

# -----------------------------
# Record audio
# -----------------------------
def record_audio(duration=5, fs=16000):
    print("Listening...")
    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    return audio.flatten(), fs

# -----------------------------
# Ask AI
# -----------------------------
def ask_ai(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# -----------------------------
# Main Program
# -----------------------------
print("AI Robot started. Speak your question...")
speak("Hello, I am ready. Ask me anything.")

try:
    while True:
        audio_data, fs = record_audio()

        audio = sr.AudioData(
            audio_data.tobytes(),
            fs,
            2
        )

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)

            if text.lower() in ["stop", "exit", "quit"]:
                speak("Stopping the program. Goodbye.")
                break

            answer = ask_ai(text)
            speak(answer)

        except sr.UnknownValueError:
            print("Could not understand")
            speak("Sorry, please repeat.")

except KeyboardInterrupt:
    print("\nStopped")
    speak("Program stopped")
