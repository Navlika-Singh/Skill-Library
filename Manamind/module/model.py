from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", max_length=8192, task=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        self.max_length = max_length
        self.task = task or "Given a skill description query, retrieve relevant skills that match the query"

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def add_instruction(self, query):
        return f"Instruct: {self.task}\nQuery:{query}"

    def tokenize(self, texts):
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return batch.to(self.model.device)

    def embed(self, texts):
        batch = self.tokenize(texts)
        with torch.no_grad():
            outputs = self.model(**batch)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.float().cpu().numpy()  # cast to float32 before numpy conversion

    def encode_query(self, queries):
        instructed = [self.add_instruction(q) for q in queries]
        return self.embed(instructed)

    def encode_document(self, documents):
        return self.embed(documents)

    def similarity(self, queries, documents):
        query_embeddings = self.encode_query(queries)
        document_embeddings = self.encode_document(documents)
        return query_embeddings @ document_embeddings.T

class LLM:
    def __init__(self, model_name="Qwen/Qwen3-4B", device_map="auto", torch_dtype="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        self.think_token = 151668  # </think>

    def generate(self, prompt, enable_thinking=True, max_new_tokens=32768):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        if enable_thinking:
            try:
                index = len(output_ids) - output_ids[::-1].index(self.think_token)
            except ValueError:
                index = 0
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return thinking_content, content
        else:
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            return None, content