# 📎 CLIP: Contrastive Language–Image Pretraining

OpenAI's **CLIP (Contrastive Language–Image Pretraining)** is a powerful multi-modal model that connects vision and language. It enables zero-shot classification, retrieval, and other cross-modal tasks by learning from natural language supervision.

---

## 🧠 Summary of CLIP

**CLIP** jointly trains an image encoder and a text encoder to predict which images and texts go together. By leveraging **natural language descriptions** instead of manual labels, CLIP learns rich visual and textual representations aligned in the same latent space.

CLIP can:
- Perform **zero-shot image classification** with prompts.
- Be used for **image-text retrieval** and **cross-modal understanding**.
- Generalize across a wide range of vision tasks without task-specific finetuning.

---

## 📄 Key Papers & Summaries

### 🔹 [CLIP: Learning Transferable Visual Models From Natural Language Supervision (2021)](https://arxiv.org/abs/2103.00020)
**Authors**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, et al.  
**Summary**: Introduces CLIP, a dual-encoder model trained on 400M image-text pairs. Demonstrates strong performance on over 30 benchmarks without task-specific training.

---

## 🧩 How CLIP Works

- **Architecture**: Two separate encoders — one for images (e.g., ViT, ResNet) and one for text (Transformer).
- **Training Objective**: Contrastive loss (InfoNCE) between matching image-text pairs.
- **Dataset**: 400 million image–text pairs collected from the internet.
- **Zero-shot Inference**: Uses natural language prompts like "a photo of a cat" to classify images.

---

## 📌 Key Points to Keep in Mind

- 📊 **Zero-shot performance**: Can perform better than supervised models in some tasks.
- 🏷️ **Prompt engineering matters**: Carefully worded prompts significantly affect performance.
- 🌍 **Web data bias**: Learned representations may reflect social, cultural, or harmful biases from training data.
- ⚙️ **Encoder flexibility**: Supports various backbone architectures (ResNet, ViT).
- 🧠 **Transferable features**: CLIP features work well in downstream vision-language applications.

---

## 🎥 Videos

- [CLIP Explained Visually – Yannic Kilcher](https://www.youtube.com/watch?v=I5uF0p73QM8)
- [CLIP - Connecting Text and Images - AI Ephiphany](https://www.youtube.com/watch?v=fQyHEXZB-nM)


---

## 📚 Related Articles & Resources

- [OpenAI CLIP Announcement](https://openai.com/research/clip)
- [Hugging Face CLIP Models](https://huggingface.co/models?search=clip)
- [CLIP in PyTorch – OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Prompt Engineering for CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb)

---

## 🚀 Quickstart (Hugging Face Transformers)
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("your-image.jpg")
text = ["a photo of a dog", "a photo of a cat"]

inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)
```

---

## 🔬 Use Cases

- Zero-shot classification
- Image search and retrieval
- Multi-modal embeddings
- Cross-modal understanding
- Prompt-based vision tasks

---

> 🧾 *CLIP is a foundational model in multi-modal AI. Its simplicity and flexibility make it a cornerstone for building vision-language applications.*

---

**Contributions welcome!** 🤝  
For improvements, issues, or questions, feel free to open a discussion or PR.