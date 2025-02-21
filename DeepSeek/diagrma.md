![image](https://github.com/user-attachments/assets/9cc6e098-1efa-4b88-ba3e-6b254a5650fe)

Feature	LoRA (Low-Rank Adaptation)	Prefix Tuning	Prompt Tuning	LLM Distillation
Method	Injects small trainable layers into existing weights	Adds task-specific prefixes to hidden states	Trains soft prompts (embedding vectors)	Compresses LLM into a smaller model
Parameter Efficiency	✅ Very efficient (trains fewer parameters)	✅ Efficient	✅ Efficient	❌ Less efficient (requires training a new model)
Training Complexity	Moderate (requires updating adapter layers)	Low (optimizes prefix embeddings)	Low (optimizes soft prompts)	High (requires full model training)
Inference Overhead	✅ Low	✅ Low	✅ Low	✅ Very Low
Adaptability	✅ Works well for different tasks	✅ Good for different NLP tasks	✅ Good for NLP tasks	✅ Can generalize if trained well
Fine-Tuning Scope	Trains small adapter layers	Trains a prefix for each task	Trains soft prompts for each task	Compresses knowledge into a smaller model
Storage Requirement	✅ Minimal (small adapter layers)	✅ Minimal	✅ Minimal	❌ High (new model must be stored)
Performance	⚡ High accuracy with fewer trainable parameters	⚡ Good performance, but limited flexibility	⚡ Good for specific domains	⭐ Can match original model if done well
Use Cases	Efficient fine-tuning for multiple tasks	NLP tasks like text classification, summarization	Few-shot learning, NLP-specific tasks	Reducing model size for deployment


![image](https://github.com/user-attachments/assets/902a7dff-b047-43bc-8408-d1b7fee4920a)
