import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

def speech_to_text_whisper():
    import os
    import tempfile
    from groq import Groq
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening (Whisper)...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_filename = temp_audio.name
            temp_audio.write(audio.get_wav_data())
            
        with open(temp_filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, file.read()),
                model="whisper-large-v3",
                response_format="text"
            )

        os.remove(temp_filename)
        return transcription.strip().lower()
        
    except Exception as e:
        print(f"Whisper Error: {e}")
        return None

if __name__ == "__main__":
    #print("Testing Google STT:")
    #print(f"Result: {speech_to_text()}")
    
    print("\nTesting Whisper STT:")
    print(f"Result: {speech_to_text_whisper()}")

