from transformers import pipeline

# Load a text-generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate color analysis
result = generator("Provide a color analysis for fair skin with dark brown eyes and pink lips:", max_length=100)
print(result[0]['generated_text'])
