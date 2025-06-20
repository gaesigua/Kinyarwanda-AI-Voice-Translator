#1. I have installed the gradio framework, let me now import it here.
import gradio as gr

#2. I have installed the transformers framework (for the Whisper model and processor), let me now import it here.
import transformers

#3. Let me import my Operating System, to ensure it runs smoothly
import os

#4. Let me import torch and torchaudio (for audio loading) libraries
import torch

import torchaudio

#5. Let me import the Whisper model

import whisper

#6. Later, I will need gTTS for text-to-speech

from gtts import gTTS

#7. I will need to manage temporary files for audio outputs

import tempfile

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# A. Let me create my FIRST function, which is a Voice Transcription. This function will load the audio_file_path, but before that I will load the Whisper model from OpenAi, and then transcribe the audio

# I will load the Whisper model once GLOBALLY to avoid reloading it on every function call. Choosing 'base' or 'small' is usually a good starting point for speed.
# For Kinyarwanda, 'large' might offer better accuracy but will be slower and require more RAM. We can change 'base' to 'small', 'medium', or 'large' as needed.
# My arguments/parameters is an audio_file_path of type String; this audio_file_path comes from the Gradio's microphone I defined
# The function should return a String of a transcribed text in Kinyarwanda.

WHISPER_MODEL = whisper.load_model("base")

def voice_transcription(audio_file_path):

    if audio_file_path is None:
        return "No audio input provided."
    print(f"Starting transcription for: {audio_file_path}")

    try:
        # the Whisper model will automatically handle audio loading and resampling;
        # and here I will pass in the audio_file_path to be transcribed. To help the model, I will specify the language English, though it can auto-detect.
        # For now, I am explicitly setting the language to English ("en") but this program will be specifically for Kinyarwanda language.

        result = WHISPER_MODEL.transcribe(audio_file_path, language="en") #For now I am setting the language to English, we will change it to kinyarwanda later on
        transcribed_text= result["text"]
        print(f"Transcription complete: {transcribed_text}")

        return transcribed_text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Error during transcription: {e}"


# B.    Let me create my SECOND function, which is a Text Translation.
#       This function will translate the English text (since it is what has been transcribed) to target languages (English, French, Spanish, German, and Kiswahili)
#       To achieve this I will use the NLLB-200 model from the transformers library

        #Let's load NLLB-200 model and tokenizer globally for efficiency
        # Using a distilled version for faster inference on CPU.
        NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
        NLLB_TOKENIZER = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
        NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)

        #Let's now create a pipeline for convenience

        NLLB_TRANSLATOR = pipeline(
            "translation",
            model = NLLB_MODEL,
            tokenizer = NLLB_TOKENIZER,
            src_lang = "eng_latn",             # Source language for this pipeline will be in English
            device = -1                        # -1 is for CPU, 0 is for GPU. Since GPU is not available I will use -1
        )


def text_translation():
    return ""

# Let me create my THIRD function, which is a Text to Speech. It will return None for now

def text_to_speech():
    return None


# Let me create my MAIN function, we will name it Voice_to_Voice
# the main function will receive as a parameter an audio_file


def voice_to_voice(audio_file_path):

    # the function will then perform/call the voice_transcription function

    transcription_response = voice_transcription(audio_file_path)

    # For now, let me just return a transcription to see if it is working, we will expand this to include translations and audio outputs later

    # I'll need placeholders for the 5 audio outputs expected by Gradio. For now, let me return empty strings/None, but eventually these will be file paths
    # The order must match: English, French, Spanish, German, Kiswahili

    return transcription_response, None, None, None, None, None


# -----------------Let's build the Gradio Interface---------------------------

# Let me create my input of Type Audio, that will be "entered" (sources) through our "microphone", of type "filepath", and show a download button

audio_input = gr.Audio(sources=["microphone"], type="filepath", show_download_button=True)


# Let me return the Audios in English, French, Spanish, German, and Kiswahili as my outputs

interface = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[
        gr.Textbox(label="Kinyarwanda Transcription (for testing)"),
        gr.Audio(label="English"),
        gr.Audio(label="French"),
        gr.Audio(label="Spanish"),
        gr.Audio(label= "German"),
        gr.Audio(label= "Kiswahili")
    ],
    title="KinyarwandAI Voice Translator",
    description="Speak in Kinyarwanda and get real-time translations to English, French, Spanish, German, and Kiswahili "

)

# Let me launch my layout (interface)

if __name__ == "__main__":
    interface.launch(share=True)