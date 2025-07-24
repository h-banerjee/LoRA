LoRA Fine-Tuning on MNIST
This project demonstrates parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) on an MNIST digit classifier built in PyTorch.
Instead of updating the entire network, it leverages LoRA to adapt model performance with minimal additional parameters, preserving original weights.

📌 Project Highlights
✅ Built a multi-layer neural network (MNISTClassifier) to classify handwritten digits.
✅ Implemented LoRA parameterization to introduce small trainable matrices that adjust model outputs.
✅ Fine-tuned the model only on digit 9 images using LoRA while keeping original weights frozen.
✅ Verified that:

Original weights remain unchanged post fine-tuning.

LoRA significantly improves digit 9 classification with <0.25% increase in total parameters.
✅ Evaluated model performance with and without LoRA to measure adaptation effectiveness.

🛠️ Technologies Used
Python

PyTorch

torchvision

tqdm

Matplotlib

💡 What is LoRA?
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that introduces low-rank matrices to adapt large pretrained models without modifying their original weights.
This drastically reduces training costs and preserves base model capabilities for multi-task or domain adaptation.

⚙️ How it works here
Train baseline model on MNIST.

Evaluate poor digit 9 performance.

Introduce LoRA modules to each linear layer’s weights.

Freeze base model weights, train only LoRA parameters on digit 9.

Verify:

Original weights are untouched.

Digit 9 classification accuracy improves significantly with LoRA enabled.

📈 Results
Metric	Baseline	LoRA Fine-Tuned
Overall Accuracy	~95.3%	~84.3% (overfit to digit 9)
Digit 9 Wrong Predictions	107	11

🔎 Observation: Fine-tuning only on digit 9 improves its classification drastically but slightly reduces generalization.

🚀 How to run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/lora-mnist.git
cd lora-mnist
Install dependencies:

bash
Copy
Edit
pip install torch torchvision tqdm matplotlib
Run the script:

bash
Copy
Edit
python main.py
