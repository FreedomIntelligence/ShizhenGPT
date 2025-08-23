import argparse
import os
import torch
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
from threading import Thread
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    TextIteratorStreamer
)
from qwen_vl_utils import fetch_image
from copy import deepcopy

# Argument parsing for model path
def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-modal chatbot")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model")
    return parser.parse_args()

# Load the model
def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:1", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    processor.chat_template = processor.tokenizer.chat_template
    return model, processor

# Audio processing
def process_audio(audio):
    if audio is None:
        return None
    try:
        sr, y = audio
        y = y[:, 0] if y.ndim > 1 else y
        save_path = "./temp.wav"
        sf.write(save_path, y, sr)
        y_resampled = librosa.load(save_path, sr=processor.feature_extractor.sampling_rate)[0]
        return y_resampled
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Streaming response generation
def generate_with_streaming(model, processor, text, images=None, audios=None, history=None):

    processed_images = []
    if images is not None and images:
        text = ''.join(['<|vision_start|><|image_pad|><|vision_end|>']*len(images)) +  text
        processed_images = [fetch_image({"type": "image", "image": img, "max_pixels": 360*420}) 
                            for img in images if img is not None]
    else:
        processed_images = None
    
    processed_audios = []
    if audios is not None and audios:
        text = ''.join(['<|audio_bos|><|AUDIO|><|audio_eos|>']*len(audios)) +  text
        processed_audios = [audio for audio in audios if audio is not None]
    else:
        processed_audios = None
    
    messages = []
    if history:
        for user_msg, assistant_msg in history:
            messages.append({'role': 'user', 'content': user_msg})
            if len(assistant_msg) > 0:  # 确保助手消息不为空
                messages.append({'role': 'assistant', 'content': assistant_msg})
    
    for xx in messages:
        xx['content'] = xx['content'].replace('<|audio_bos|><|AUDIO|><|audio_eos|>', '').replace('<|vision_start|><|image_pad|><|vision_end|>', '')
    messages.append({'role': 'user', 'content': text})
    
    print('messages',messages,flush=True)
    print('processed_images',processed_images,flush=True)
    print('processed_audios',processed_audios,flush=True)
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not text:
        text = [""]

    input_data = processor(
        text=[text],
        audios=processed_audios,
        images=processed_images, 
        return_tensors="pt", 
        padding=True
    )
    print('input_ids',processor.tokenizer.decode(input_data['input_ids'][0]),flush=True)
    
    for k, v in input_data.items():
        if hasattr(v, "to"):
            input_data[k] = v.to(model.device)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True,skip_prompt=True)
    generation_kwargs = dict(
        **input_data,
        streamer=streamer,
        max_new_tokens=1500,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for new_text in streamer:
        yield new_text

def predict(message, image, audio, chatbox):

    chat_history = deepcopy(chatbox)
    print('[chat_history]',chat_history,flush=True)
    print('[message]',message,flush=True)
    
    processed_audio = None
    if audio is not None:
        processed_audio = [process_audio(audio)]
    
    processed_image = None
    if image is not None:
        processed_image = [image]

    chatbox.append([message, ""])
    response = ""
    
    for chunk in generate_with_streaming(model, processor, message, processed_image, processed_audio, chat_history):
        response += chunk
        chatbox[-1][1] = response
        yield chatbox
    
    print("\n=== Complete Model Response ===")
    print(response)
    print("============================\n", flush=True)
    
    return chatbox

# CSS for Gradio interface
css = """
.gradio-container {
    background-color: #f7f7f7;
    font-family: 'Arial', sans-serif;
}
.chat-message {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.user-message {
    background-color: #e6f7ff;
    border-left: 5px solid #1890ff;
}
.bot-message {
    background-color: #f2f2f2;
    border-left: 5px solid #52c41a;
}
.title {
    text-align: center;
    color: #1890ff;
    font-size: 24px;
    margin-bottom: 20px;
}
"""

# Gradio UI setup
def setup_gradio_interface():
    with gr.Blocks(css=css) as demo:
        gr.HTML("<h1 class='title'>TCM-Omni 中医多模态大模型（融合望闻问切）</h1>")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500)
                message = gr.Textbox(label="Input your question", placeholder="Please type your question here...")

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image")
                audio_input = gr.Audio(type="numpy", label="Record or Upload Audio")

        submit_btn.click(predict, inputs=[message, image_input, audio_input, chatbot], outputs=[chatbot], show_progress=True).then(
            lambda: "", outputs=[message]
        )

        clear_btn.click(lambda: (None, None, None, []), outputs=[message, image_input, audio_input, chatbot])

    demo.queue().launch(server_name="0.0.0.0", server_port=7862, share=True)

# Main entry point
if __name__ == "__main__":
    args = parse_args()
    model, processor = load_model(args.model_path)
    setup_gradio_interface()


