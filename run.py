import gradio as gr
import openai
import whisper
from dotenv import load_dotenv
import os
import warnings


load_dotenv()

# Ignore all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

openai.api_key = os.getenv("OPENAI_API_KEY")
messages = [
    {"role": "system", "content": "You are a Professor of Linguistics at Oslo University, your speciality is accents and old norwegian language. You are instructed to help transcribe some text. When you transcribe you put strong emphasis on maintaining the original text and do not make changes but look specifically at spelling mistakes and obvious errors. Return the original version as well as your corrected one. Also add a summary of changes made and the reason plus some interesting facts about the text."}
]


def speech_to_text(tmp_filename, user_message):
    global messages
    
    #check if we should process a file?
    if (tmp_filename):
        model = whisper.load_model("large-v2")
        transcription = model.transcribe(
            tmp_filename, language="no", temperature=0.0)
        user_message = transcription["text"]

    #either the file transcript or chat-message from here on.

    # debug info
    print("Input:", user_message)
    
    
    # Do the GPTChat processing of input.
    messages.append({"role": "user", "content": user_message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # latest conversation mesg.
    AImessage = response["choices"][0]["message"]["content"]
    
    #debug info
    print("AI output:", AImessage)

    # complete chat history
    messages.append({"role": "assistant", "content": AImessage})
    chat = ''
    for message in messages:
        if message["role"] != 'system':
            chat += message["role"] + ':' + message["content"] + "\n\n"

    return (user_message, chat, AImessage)


#gradio inputs
input_text = gr.inputs.Textbox(lines=4, label="Text Chat")

# gradio UI
ui = gr.Interface(
    fn=speech_to_text,
    inputs=[
        gr.Audio(source="microphone", type="filepath", label="Audio"),
        input_text],
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="Chat history", ),
        gr.Textbox(label="Chat response")]
)


ui.launch()
