from PIL import Image
from typing import Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import json
import re

from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

# Configurations Constants
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 1

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
            max_new_tokens=16,
            do_sample=False
        )
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Initializing MyAgent")

    def get_batch_size(self) -> int:
        return BATCH_SIZE

    def get_image_information(self, images, queries):
        preprocessed_images_info = []

        # Get And Preprocess Info
        for index, image in enumerate(images):
            count = 1
            while True:
                api_result = self.search_pipeline(image, k=count)[count-1]

                entity_info = {}
                entity = api_result["entities"][0]
                entity_info["entity_name"] = entity["entity_name"]

                if entity["entity_attributes"] == None:
                    continue

                for key, value in entity["entity_attributes"].items():
                    value = str(value)

                    # non ASCII char
                    value = re.sub(r'[^\x00-\x7F]+', '', value)

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
                
                    entity_info[key] = value
                
                entity_string = f"name={entity['entity_name']}"
                for key, value in entity.items():
                    if key != "entity_name":
                        entity_string += f" {key}={value}"
                
                query_emb = self.semantic_model.encode(queries[index], convert_to_tensor=True)
                entity_emb = self.semantic_model.encode(entity_string, convert_to_tensor=True)
                
                scores = util.cos_sim(query_emb, entity_emb)[0]
                if scores >= 0.7:
                    preprocessed_images_info.append(entity_info)
                    break

        return preprocessed_images_info

    def generate_answer(self, images_info, queries) -> list[str]:
        prompts = []

        for image_info, query in zip(images_info, queries):
            prompt = f"""
                    You are a helpful assistant which answers user question based on the given information below,
                    keep the answers as short and simple as possible, if you dont know say I don't know.

                    Information:
                    {json.dumps(image_info, ensure_ascii=False, indent=3)}

                    User Question:
                    {query}

                    Assistant: 
                    """
            prompts.append(prompt)
        
        outputs = self.generator(prompts)
        responses = [output[0]["generated_text"].split("Assistant: ")[-1].strip() for output in outputs]
        
        for output in outputs:
            print(output[0]["generated_text"].strip(), end="\n\n")
        
        return responses

    def batch_generate_response(
        self,
        queries: list[str],
        images: list[Image.Image],
        message_histories: list[list[dict[str, Any]]] = None,
        ) -> list[str]:
        
        images_info = self.get_image_information(images, queries)
        
        responses = self.generate_answer(images_info, queries)

        return responses