"""Answer generation module using Qwen/Qwen3-0.6B.

Generates natural language answers grounded strictly in retrieved context snippets.
"""

from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .chunking import read_file
from .models import MinimalSource


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


class AnswerGenerator:
    """Answers questions based on retrieved context using Qwen3 causal language model."""

    _instance: Optional["AnswerGenerator"] = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Initialize generator with specified model name and CPU optimization."""
        self.model_name = model_name
        # Optimize CPU threads for inference
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL_NAME) -> "AnswerGenerator":
        """Get or initialize singleton instance of AnswerGenerator."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def extract_context(
        self, sources: List[MinimalSource], max_chars: int = 1800, max_snippets: int = 3
    ) -> str:
        """Read and format text content for top retrieved sources."""
        file_cache: dict[str, str] = {}
        snippets: List[str] = []
        total_len = 0

        for src in sources[:max_snippets]:
            if src.file_path not in file_cache:
                file_cache[src.file_path] = read_file(src.file_path)
            content = file_cache[src.file_path]
            snippet = content[src.first_character_index:src.last_character_index].strip()
            if snippet:
                if total_len + len(snippet) > max_chars:
                    budget = max(0, max_chars - total_len)
                    if budget > 100:
                        snippets.append(snippet[:budget])
                    break
                snippets.append(snippet)
                total_len += len(snippet)

        return "\n\n".join(snippets)

    def generate_answer(
        self, question: str, sources: List[MinimalSource], max_new_tokens: int = 60
    ) -> str:
        """Generate a concise answer to the question using retrieved sources as context."""
        context = self.extract_context(sources)
        if not context:
            return "No relevant context found in codebase to answer this question."

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer the question directly and concisely in 1-2 sentences "
            "based on the context above:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean prompt markers and duplicate prefixes
        while answer.lower().startswith("answer:"):
            answer = answer[len("answer:"):].strip()
        if "\n\n" in answer:
            answer = answer.split("\n\n")[0].strip()
        if "\nQuestion:" in answer:
            answer = answer.split("\nQuestion:")[0].strip()
        if "Context:" in answer:
            answer = answer.split("Context:")[0].strip()

        return answer or "Unable to derive a definitive answer from the provided context."
