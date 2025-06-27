from PIL import Image
from typing import Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import re

from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

# Configurations Constants
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 1
SEARCH_RESULTS = 5

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
    
    def preprocess_image_information(self, images_information):
        results = []

        print(json.dumps(images_information, indent="\t"))
        for information in images_information:
            result = {}
            for output in information:
                for entity in output["entities"]:
                    result["entity_name"] = entity["entity_name"]

                    for key, value in entity["entity_attributes"].items():
                        # HTML Tags
                        value = re.sub(r'<.*?>', '', value)

                        # Whitespace Char
                        value = re.sub(r'[\n\t\r]', '', value).strip()

                        # Wikipedia Template
                        value = re.sub(
                            r'\{\{([^\{\}]+)\}\}',
                            lambda m: ' '.join([p for p in m.group(1).split('|')[1:] if '=' not in p]),
                            value
                        )

                        # Wikipedia Links
                        value = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r' \2', value)
                        value = re.sub(r'\[\[([^\]]+)\]\]', r' \1', value)

                        result[key] = value.strip()
            results.append(result)
        
        return results

    def get_batch_image_info(self, images: list[Image.Image]):
        image_search_results_batch = []

        for image in images:
            image_search_results_batch.append(self.search_pipeline(image, k=SEARCH_RESULTS))

        return image_search_results_batch

    def batch_generate(self, images_information, user_queries) -> list[str]:
        prompts = []

        for image_info, query in zip(images_information, user_queries):
            prompt = f"""
                    You are a helpful assistant which answers user question based on the given information below,If you dont know say I don't know.

                    Information:
                    {json.dumps(image_info, ensure_ascii=False)}

                    User Question:
                    {query}

                    Assistant: 
                    """
            prompts.append(prompt)
        
        outputs = self.generator(prompts)
        responses = [output[0]["generated_text"].split("Assistant: ")[-1].strip() for output in outputs]
        print(responses)
        
        return responses

    def batch_generate_response(
        self,
        queries: list[str],
        images: list[Image.Image],
        message_histories: list[list[dict[str, Any]]] = None,
        ) -> list[str]:
        
        images_information = self.get_batch_image_info(images)
        images_information = self.preprocess_image_information(images_information)
        print(images_information)
        
        responses = self.batch_generate(images_information, queries)

        return responses