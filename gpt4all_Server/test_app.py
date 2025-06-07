from gpt4all import GPT4All

MODEL_DIR = r"C:\Users\ahmad\AppData\Local\nomic.ai\GPT4All"
MODEL_NAME = "Llama-3.2-1B-Instruct-Q4_0.gguf"

try:
    model = GPT4All(model_name=MODEL_NAME, model_path=MODEL_DIR)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

prompt = "Explain artificial intelligence simply like you're talking to a child"
output = model.generate(prompt=prompt, max_tokens=300, temp=0.7, top_k=40, n_batch=512)

print("Generated text:\n", output)
