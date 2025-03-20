# Transformers: The Revolutionary Deep Learning Model

Transformers have revolutionized Natural Language Processing (NLP) and various other domains with their self-attention mechanism. Introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762), Transformers have become the foundation of modern AI architectures like BERT, GPT, and T5.

---

## 📜 Paper
[Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)

---

## 📖 Summary
Transformers introduce a self-attention mechanism that allows models to weigh the importance of different words in a sentence, regardless of their position. Unlike RNNs, Transformers process input data in parallel, significantly improving computational efficiency.

### 🔹 Core Components:
- **Self-Attention**: Computes attention scores to capture long-range dependencies.
- **Positional Encoding**: Adds information about the order of tokens.
- **Multi-Head Attention**: Enables the model to focus on different parts of the input.
- **Feed-Forward Networks**: Fully connected layers for transformation.
- **Layer Normalization**: Stabilizes training and improves convergence.

---

## 📌 Key Points to Keep in Mind
- No recurrence or convolution; relies entirely on self-attention.
- Parallelized computation leads to faster training.
- Essential for NLP tasks like translation, summarization, and question-answering.
- Forms the backbone of large-scale AI models like GPT, BERT, and T5.
- Requires substantial computational resources for training.

---

## 🎥 Video Explanation
[**Transformers Explained**](https://www.youtube.com/watch?v=4Bdc55j80l8) - Yannic Kilcher<br>
[**StatsQuest: Transformers Neural Network**](https://www.youtube.com/watch?v=zxQyTK8quyY)<br>
[**StatsQuest: Encoder-Only Transformers**](https://www.youtube.com/watch?v=GDN649X_acE&t=3s)<br>
[**StatsQuest: Decoder-Only Transformers**](https://www.youtube.com/watch?v=bQ5BoolX9Ag&t=1731s)<br>
[**StatsQuest: Understand mathematics of Queries, Keys, Values**](https://www.youtube.com/watch?v=KphmOJnLAdI&t=268s)<br>


---

## 📚 Related Articles
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Exploring Transformers](https://huggingface.co/course/chapter1)

---

## 🚀 Get Started with Transformers
You can experiment with Transformer models using the Hugging Face `transformers` library:
```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "Transformers have changed NLP forever."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

---

### 🌟 Contributors & References
- Paper: ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- Hugging Face Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- OpenAI GPT, Google BERT, T5


🔹 *Feel free to contribute and improve this documentation!*
