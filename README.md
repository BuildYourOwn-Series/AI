# Build Your Own AI

Companion code for *Build Your Own AI*: AI models, transformers, and a fully local C++ chatbot built from first principles.

This repository contains the complete, minimal, and fully-explained implementations that accompany the book.  
Every algorithm is presented in clean, readable code — without hidden frameworks or opaque abstractions — so you can understand how modern AI systems really work.

---

## 📘 What’s Inside

The repo is organized to follow the progression of the book:

### **1. Symbolic Foundations**
- Rule-based chatbots  
- Pattern matching and unification  
- Prolog-style inference  

### **2. Numerical & Neural Methods**
- Perceptrons and activation functions  
- Multilayer neural networks (MLPs)  
- Gradient descent and backpropagation  
- Softmax classifiers  

### **3. Representations**
- Embeddings  
- Vector semantics  
- Similarity search  

### **4. Transformers, Attention & Modern Deep Learning**
- Scaled dot-product attention  
- Multi-head attention  
- Positional encodings  
- A fully working TinyTransformer  

### **5. Quantization & Inference**
- INT8 quantization  
- Model compression  
- Fast inference paths  

### **6. Putting It All Together**
- A complete local chatbot in **C++**  
- Retrieval-based architecture  
- Fast embedding search  
- Instruction-following behavior without internet or cloud APIs  

Everything is designed to run locally, reproducibly, and with minimal dependencies.

---

## 🧱 Directory Structure

```text
AI/
├── chapX/           # Code listings found in each chapter
├── models/          # Where your models will go
├── mnist/           # Placeholder where you'll place MNIST
````

The repository intentionally excludes large datasets (e.g., MNIST) and includes only small, book-compatible samples.

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/BuildYourOwn-Series/AI.git
cd AI
```

Each chapter can be run and compiled according to instructions in the book (C++ requires a C++20 compiler):

Python components (if used in chapters) can be run directly:

```bash
cd chap01
python3 tictactoe.py
```

---

## 📚 About the Book

This repository accompanies **Build Your Own AI** by **Frederick von Mame**.

The book explains AI the way engineers learn best:

* minimal examples
* no magic
* one transparent piece at a time
* everything runnable

If you have the book: this repo matches chapter numbers and code listings.
If you don’t: you can still use the code as a hands-on crash course in modern AI systems.

---

## 🧩 License

All code in this repository is released under the **MIT License**, unless otherwise noted.
Example datasets are small, open, or generated for demonstration purposes.

---

## 🤝 Contributing

While the series is tightly authored, issues and suggestions are welcome.
If you find a typo, bug, or optimization opportunity, open an issue or PR.

---

## ✨ Philosophy

> **Understanding comes from building.**
> To truly learn AI, you must create it yourself — one line at a time.
