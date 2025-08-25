# Towards Multimodal LLMs for Traditional Chinese Medicine

<div align="center">
<h3>
  ShizhenGPT
</h3>
</div>

<div align="center">
<h4>
  📃 <a href="https://arxiv.org/abs/2508.14706" target="_blank">Paper</a> ｜ 📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT" target="_blank">TCM Pre-training Dataset</a> | 📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT" target="_blank">TCM Instruction Data</a> | 📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/TCM-Text-Exams" target="_blank">TCM Text Benchmark</a>
</h4>
  <h4>
  🤗 <a href="https://huggingface.co/FreedomIntelligence/ShizhenGPT-7B-Omni" target="_blank">ShizhenGPT-7B</a> | 🤗 <a href="https://huggingface.co/FreedomIntelligence/ShizhenGPT-32B-VL" target="_blank">ShizhenGPT-32B</a>
</h4>
</div>

## ⚡ Introduction
Hello! Welcome to the repository for [ShizhenGPT](https://arxiv.org/abs/2508.14706)! 

<div align=center>
<img src="assets/image.png"  width = "50%" alt="ShizhenGPT" align=center/>
</div>

**ShizhenGPT** is the first multimodal LLM designed for Traditional Chinese Medicine (TCM). Trained extensively, it excels in TCM knowledge and can understand images, sounds, smells, and pulses (支持望闻问切).

## 📚 The Largest Open-Source TCM Dataset

We open-source the largest available TCM dataset, consisting of a pretraining dataset and an instruction fine-tuning dataset.

|                             | Quantity    | Description                                                                                       | Download Link                                                                                                                                  |
| --------------------------- | ----------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **TCM Pretraining Dataset** | \~6B tokens | Injects TCM knowledge and aligns visual and auditory understanding. | [FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT](https://huggingface.co/datasets/FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT)           |
| **TCM Instruction Dataset** | 27K items   | Fine-tunes TCM LLMs to improve instruction-following and response quality.  | [FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT](https://huggingface.co/datasets/FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT) |


## 👨‍⚕️ Model

#### Model Access

> **ShizhenGPT-7B** is available on Huggingface:

|                        | Parameters | Supported Modalities          | Link                                                                  |
| ---------------------- | ---------- | ----------------------------- | --------------------------------------------------------------------- |
| **ShizhenGPT-7B-LLM**  | 7B         | Text                          | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-7B-LLM) |
| **ShizhenGPT-7B-VL**   | 7B         | Text, Image Understanding     | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-7B-VL) |
| **ShizhenGPT-7B-Omni** | 7B         | Text, Four Diagnostics (望闻问切) | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-7B-Omni) |
| **ShizhenGPT-32B-LLM**  | 32B        | Text                          | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-32B-LLM) |
| **ShizhenGPT-32B-VL**   | 32B        | Text, Image Understanding     | [HF Link](https://huggingface.co/FreedomIntelligence/ShizhenGPT-32B-VL) |
| **ShizhenGPT-32B-Omni** | 32B        | Text, Four Diagnostics (望闻问切) | Available soon                                                          |

*Note: The LLM and VL models are parameter-split variants of ShizhenGPT-7B-Omni. Since their architectures align with Qwen2.5 and Qwen2.5-VL, they are easier to adapt to different environments. In contrast, ShizhenGPT-7B-Omni requires `transformers==0.51.0`.*


#### Model Inference

<details open>
<summary><h4>A. Launch with Gradio Demo</h4></summary>

```shell
pip install gradio
python demo/app_shizhengpt.py --model_path FreedomIntelligence/ShizhenGPT-7B-Omni
```

</details>

<details open>
<summary><h4>B. Text-based Inference</h4></summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("FreedomIntelligence/ShizhenGPT-7B-LLM",torch_dtype="auto",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("FreedomIntelligence/ShizhenGPT-7B-LLM")

input_text = "为什么我总是手脚冰凉，是阳虚吗？"
messages = [{"role": "user", "content": input_text}]

inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True
), return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</details>


<details open>

<summary><h4>C. Image-Text-to-Text</h4></summary>

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


processor = AutoProcessor.from_pretrained("FreedomIntelligence/ShizhenGPT-7B-VL")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("FreedomIntelligence/ShizhenGPT-7B-VL", torch_dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/path/to/your/image.png",
            },
            {"type": "text", "text": "请从中医角度解读这张舌苔。"},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

</details>

<details open>

<summary><h4>D. Signal-Image-Text-to-Text</h4></summary>

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import fetch_image
import librosa

# Load model and processor
model_path = 'FreedomIntelligence/ShizhenGPT-7B-Omni'
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto").cuda()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def generate(text, images=None, signals=None):
    # Process images if provided
    processed_images = []
    if images is not None and images:
        text = ''.join(['<|vision_start|><|image_pad|><|vision_end|>']*len(images)) + text
        processed_images = [fetch_image({"type": "image", "image": img, "max_pixels": 360*420}) 
                            for img in images if img is not None]
    else:
        processed_images = None
    
    # Process audio signals if provided
    processed_signals = []
    if signals is not None and signals:
        text = ''.join(['<|audio_bos|><|AUDIO|><|audio_eos|>']*len(signals)) + text
        processed_signals = [librosa.load(signal, sr=processor.feature_extractor.sampling_rate)[0] 
                             for signal in signals if signal is not None]
    else:
        processed_signals = None
    
    # Prepare messages
    messages = [{'role': 'user', 'content': text}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Ensure text is non-empty
    if not text:
        text = [""]

    # Process the input data
    input_data = processor(
        text=[text],
        audios=processed_signals,
        images=processed_images, 
        return_tensors="pt", 
        padding=True
    )
    input_data = input_data.to(model.device)
    
    # Generate the output
    generated_ids = model.generate(**input_data, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_data.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

# Example usage
# Text input
print(generate('为什么我总是手脚冰凉，是阳虚吗？'))
# Image input
print(generate('请从中医角度解读这张舌苔。', images=['path_to_image']))
# Audio input
print(generate('请回答这个语音问题', signals=['path_to_audio']))
```

</details>

## 🧐 Evaluation

<details>
<summary><h4>Text Benchmark</h4></summary>
The text benchmark is composed of five sections, each compiled from different national-level TCM examinations.

|                                      | Samples                       |
| ------------------------------------ | ------------------------------ |
| 2024 TCM Pharmacist (2024年中医药剂师考试)   | 480 |
| 2024 TCM Physician (2024年中医职业医师资格考试) | 184 |
| 2024 TCM Assistant Physician (2024年中医助理职业医师资格考试) | 138|
| 2024 TCM Graduate Entrance Examination (2024年中医综合考验真题) | 147 |
| 2025 TCM Graduate Entrance Examination (2025年中医综合考验真题) | 139 |
</details>


## 📖 Citation
```
@misc{chen2025shizhengptmultimodalllmstraditional,
      title={ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine}, 
      author={Junying Chen and Zhenyang Cai and Zhiheng Liu and Yunjin Yang and Rongsheng Wang and Qingying Xiao and Xiangyi Feng and Zhan Su and Jing Guo and Xiang Wan and Guangjun Yu and Haizhou Li and Benyou Wang},
      year={2025},
      eprint={2508.14706},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.14706},
}
```
