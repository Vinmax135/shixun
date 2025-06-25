from PIL import Image
from typing import Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

# Configurations Constants
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
BATCH_SIZE = 1
SEARCH_RESULTS = 1

class MyAgent(BaseAgent):
    def __init__(self, search_pipeline: UnifiedSearchPipeline):
        super().__init__(search_pipeline)
        offload_folder = "./offload_myagent"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",           # Enables automatic offloading
            offload_folder=offload_folder,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # Do NOT set device=... here, let the model handle device placement
            max_new_tokens=8,
            do_sample=False
        )
        print("Initializing MyAgent")

    def get_batch_size(self) -> int:
        return BATCH_SIZE
    
    def get_batch_image_info(self, images: list[Image.Image]):
        image_search_results_batch = []

        for image in images:
            image_search_results_batch.append(self.search_pipeline(image, k=SEARCH_RESULTS))

        return image_search_results_batch

    def batch_generate(self, images_information, user_queries) -> list[str]:
        prompts = []

        for image_info, query in zip(images_information, user_queries):
            prompt = f"""
                    You are a helpful assistant. Given the following information about an image, 
                    answer the user's question based on given image informations accurately 
                    and concisely as possible. Do not generate answers that is not sourced from 
                    the given image information, If you do not know the answer, respond with 'I don't know'.

                    Image Information:
                    {image_info}

                    User Question:
                    {query}
                    """
            prompts.append(prompt)
        
        outputs = self.generator(prompts)
        responses = [output[0]["generated_text"] for output in outputs]
        
        return responses

    def batch_generate_response(
        self,
        queries: list[str],
        images: list[Image.Image],
        message_histories: list[list[dict[str, Any]]] = None,
        ) -> list[str]:
        
        images_information = self.get_batch_image_info(images)
        responses = self.batch_generate(images_information, queries)

        return responses