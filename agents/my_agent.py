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
TOP_K = 3

class MyAgent(BaseAgent):
    def __init__(self, search_pipeline: UnifiedSearchPipeline):
        super().__init__(search_pipeline)
        offload_folder = "./offload_myagent"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
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
            max_new_tokens=16,
            do_sample=False
        )
        self.semantic_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # ✅ stronger model
        print("Initializing MyAgent")

    def get_batch_size(self) -> int:
        return BATCH_SIZE

    def get_image_information(self, images, queries):
        preprocessed_images_info = []

        for index, image in enumerate(images):
            api_results = self.search_pipeline(image, k=10)

            entities_info = []
            for api_result in api_results:
                if not api_result["entities"]:  # ✅ check if empty
                    continue

                entity_info = {}
                entity = api_result["entities"][0]
                entity_info["entity_name"] = entity["entity_name"]

                if entity["entity_attributes"] is None:
                    continue

                for key, value in entity["entity_attributes"].items():
                    value = str(value)
                    value = re.sub(r'[^\x00-\x7F]+', '', value)
                    value = re.sub(r'<.*?>', '', value)
                    value = re.sub(r'[\n\t\r]', '', value).strip()
                    value = re.sub(
                        r'\{\{([^\{\}]+)\}\}',
                        lambda m: ' '.join([p for p in m.group(1).split('|')[1:] if '=' not in p]),
                        value
                    )
                    value = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r' \2', value)
                    value = re.sub(r'\[\[([^\]]+)\]\]', r' \1', value)

                    entity_info[key] = value

                entities_info.append(entity_info)

            # Convert all entity dicts to strings for embedding
            entities_string = []
            for entity_info in entities_info:
                entity_string = f"name={entity_info['entity_name']}"
                for key, value in entity_info.items():
                    if key != "entity_name":
                        entity_string += f" {key}={value}"
                entities_string.append(entity_string)

            if not entities_string:
                preprocessed_images_info.append({"info": "No relevant entity found."})
                continue

            query_emb = self.semantic_model.encode(queries[index], convert_to_tensor=True)
            entity_emb = self.semantic_model.encode(entities_string, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, entity_emb)

            # ✅ NEW: take top-k relevant entities and merge their info
            topk_indices = scores[0].topk(min(TOP_K, len(entities_info))).indices.tolist()

            # Merge top-k entities
            merged_info = {}
            for i in topk_indices:
                entity = entities_info[i]
                for key, value in entity.items():
                    if key in merged_info:
                        if value not in merged_info[key]:
                            merged_info[key] += f"; {value}"  # avoid duplicate concat
                    else:
                        merged_info[key] = value
            preprocessed_images_info.append(merged_info)

        return preprocessed_images_info

    def generate_answer(self, images_info, queries) -> list[str]:
        prompts = []

        for image_info, query in zip(images_info, queries):
            prompt = (
                f"You are a helpful assistant. Answer the user's question based only on the info below. "
                f"If unsure, say 'I don't know'.\n\n"
                f"Info: {json.dumps(image_info, ensure_ascii=False)}\n"
                f"Question: {query}\n"
                f"Answer:"
            )
            prompts.append(prompt)

        outputs = self.generator(prompts)
        responses = [output[0]["generated_text"].split("Answer:")[-1].strip() for output in outputs]

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