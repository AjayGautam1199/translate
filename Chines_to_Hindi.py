import os
import wave
import numpy as np
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from pyAudioAnalysis import audioSegmentation as aS
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load gender classifier model
def load_gender_classifier():
    # Placeholder for loading a pre-trained model
    # Replace this with actual model loading code
    model = SVC(probability=True)
    scaler = StandardScaler()
    return model, scaler

# Classify gender of a given audio segment
def classify_gender(audio_segment, model, scaler):
    # Extract features from audio segment and classify gender
    # This is a placeholder function; replace it with actual feature extraction and classification
    features = np.random.rand(1, 10)  # Dummy features
    features = scaler.transform(features)
    gender = model.predict(features)
    return 'male' if gender == 1 else 'female'

# Perform speaker diarization
def speaker_diarization(audio_file):
    sampling_rate, signal = wavfile.read(audio_file)
    segments = aS.speakerDiarization(audio_file, 5)
    return segments

# Convert speech to text
def speech_to_text(audio_segment):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_segment) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="zh-CN")
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return None

# Translate text
def translate_text(text, src_language='zh-CN', dest_language='hi'):
    try:
        translator = Translator()
        translated = translator.translate(text, src=src_language, dest=dest_language)
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
    return None
#text
# Text-to-speech with gender-specific voices
def text_to_speech(text, gender, output_file):
    try:
        tts = gTTS(text=text, lang='hi', tld='com.in' if gender == 'female' else 'co.in')
        temp_file = f"temp_{gender}.mp3"
        tts.save(temp_file)
        sound = AudioSegment.from_mp3(temp_file)
        sound.export(output_file, format="mp3")
        os.remove(temp_file)
    except Exception as e:
        print(f"Text-to-Speech error: {e}")
        
# Main function to process audio file
def translate_audio(input_audio, output_audio):
    # Perform speaker diarization
    segments = speaker_diarization(input_audio)
    model, scaler = load_gender_classifier()

    combined_output = AudioSegment.empty()
    for i, segment in enumerate(segments):
        start, end, speaker_label = segment
        segment_audio = input_audio[start:end]

        # Classify gender
        gender = classify_gender(segment_audio, model, scaler)

        # Convert speech to text
        chinese_text = speech_to_text(segment_audio)
        if chinese_text:
            hindi_text = translate_text(chinese_text)
            if hindi_text:
                # Convert text to speech with gender-specific voice
                segment_output_file = f"segment_{i}.mp3"
                text_to_speech(hindi_text, gender, segment_output_file)
                combined_output += AudioSegment.from_mp3(segment_output_file)
                os.remove(segment_output_file)

    combined_output.export(output_audio, format="mp3")
    print(f"Translated audio saved as {output_audio}")

input_audio = "audio.mp3"
output_audio = "translate_audio.mp3"
translate_audio(input_audio, output_audio)
