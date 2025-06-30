from typing import Dict, List, Any
import os

import torch
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

from crag_web_result_fetcher import WebSearchResult
from transformers import AutoModelForVision2Seq, AutoProcessor

# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8
MAX_GENERATION_TOKENS = 75
NUM_SEARCH_RESULTS = 3

class SimpleRAGAgent(BaseAgent):
    def __init__(
        self,
        search_pipeline: UnifiedSearchPipeline,
        model_name: str = "Qwen/Qwen-VL-Chat-7B",
        max_gen_len: int = 64
    ):
        super().__init__(search_pipeline)

        if search_pipeline is None:
            raise ValueError("Search pipeline is required for RAG agent")

        self.model_name = model_name
        self.max_gen_len = max_gen_len

        self.initialize_models()

    def initialize_models(self):
        print(f"Initializing {self.model_name} with transformers...")
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm = AutoModelForVision2Seq.from_pretrained(self.model_name, trust_remote_code=True).eval().cuda()
        print("Models loaded successfully")

    def get_batch_size(self) -> int:
        return AICROWD_SUBMISSION_BATCH_SIZE

    def batch_summarize_images(self, images: List[Image.Image]) -> List[str]:
        summarize_prompt = "Please summarize the image with one sentence that describes its key elements."

        prompts = [summarize_prompt] * len(images)
        inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.llm.device)

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        summaries = self.processor.batch_decode(outputs, skip_special_tokens=True)
        summaries = [summary.strip() for summary in summaries]

        print(f"Generated {len(summaries)} image summaries")
        return summaries

    def prepare_rag_enhanced_inputs(
        self,
        queries: List[str],
        images: List[Image.Image],
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        search_results_batch = []
        search_queries = [f"{query} {summary}" for query, summary in zip(queries, image_summaries)]

        for search_query in search_queries:
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            search_results_batch.append(results)

        inputs = []
        for query, image, message_history, search_results in zip(queries, images, message_histories, search_results_batch):
            SYSTEM_PROMPT = (
                "You are a helpful assistant that truthfully answers user questions about the provided image. "
                "Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'."
            )

            rag_context = ""
            if search_results:
                rag_context = "Here is some additional information that may help you answer:\n\n"
                for i, result in enumerate(search_results):
                    result = WebSearchResult(result)
                    snippet = result.get('page_snippet', '')
                    if snippet:
                        rag_context += f"[Info {i+1}] {snippet}\n\n"

            # Compose the full prompt text combining system prompt, RAG context, history, and query
            prompt_parts = [SYSTEM_PROMPT]

            if rag_context:
                prompt_parts.append(rag_context)

            if message_history:
                # Flatten message history into text lines (assumes dict with 'role' and 'content')
                for msg in message_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    prompt_parts.append(f"{role.capitalize()}: {content}")

            prompt_parts.append(f"User: {query}")
            prompt_parts.append("Assistant:")

            full_prompt = "\n".join(prompt_parts)

            inputs.append({
                "image": image,
                "prompt": full_prompt
            })

        return inputs

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        print(f"Processing batch of {len(queries)} queries with RAG")

        # Step 1: Batch summarize images
        image_summaries = self.batch_summarize_images(images)

        # Step 2: Prepare inputs
        rag_inputs = self.prepare_rag_enhanced_inputs(
            queries, images, image_summaries, message_histories
        )

        # Prepare batched inputs for generation
        batch_images = [item["image"] for item in rag_inputs]
        batch_prompts = [item["prompt"] for item in rag_inputs]

        inputs = self.processor(
            images=batch_images,
            text=batch_prompts,
            return_tensors="pt",
            padding=True
        ).to(self.llm.device)

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=MAX_GENERATION_TOKENS,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
        responses = [response.strip() for response in responses]

        print(f"Successfully generated {len(responses)} responses")
        return responses
