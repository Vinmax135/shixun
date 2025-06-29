import torch
import re
import cv2
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from groundingdino.util.inference import load_model, predict
from agents.base_agent import BaseAgent

# Constants
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 1
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
SEARCH_COUNT = 10
TOP_K = 3

class MyAgent(BaseAgent):
    def __init__(self, search_pipeline):
        super().__init__(search_pipeline)
        
        # Load Visual Model
        config_path = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weight_path = "../GroundingDINO/groundingdino_swint_ogc.pth"
        self.visual_model = load_model(config_path, weight_path)

        # Load LLM
        offload_folder = "./offload_myagent"
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="auto",
            offload_folder=offload_folder,
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            trust_remote_code=True,
        )
        self.llm = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=16,
            do_sample=False
        )

        # Model For Semantic Relationships
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def get_batch_size(self) -> int:
        return BATCH_SIZE
    
    def crop_images(self, image, query):
        image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)    

        boxes, logits, phrases = predict(
            model=self.visual_model,
            image=image,
            caption=query,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        _, W, H = image.size()
        boxes_px = boxes.clone()

        for index in range(3):
            size = W if index % 2 == 0 else H
            boxes_px[:, index] *= size

        x0, y0 = [(boxes_px[:, 0] - boxes_px[:, 2] / 2).item(), (boxes_px[:, 1] - boxes_px[:, 3] / 2).item()]
        x1, y1 = [(boxes_px[:, 0] + boxes_px[:, 2] / 2).item(), (boxes_px[:, 1] + boxes_px[:, 3] / 2).item()]

        return image.crop((x0, y0, x1, y1))

    def extract_object(self, query):
        prompt = (
            "Extract all real-world objects (like physical things) mentioned in this query. "
            "Return them as a comma-separated list with no explanation, Do not return full sentences.\n"
            f"Query: {query}\nObjects:"
        )

        output = self.llm(prompt)[0]["generated_text"]
        result = output.split("Objects:")[-1].strip()

        # Optionally format result as 'x. y. z.'
        result = result.replace("..", ".").strip()
        if not result.endswith("."):
            result += "."

        print(result)
        return result

    def clean_metadata(self, raw_data):
        cleaned = {}
        ignored_keys = [
            "image", "image_size", "mapframe_wikidata", "coordinates",
            "website", "url", "homepage", "official_site", "wikidata", "wikibase_item"
        ]

        raw_data = raw_data[0]
        cleaned["name"] = raw_data["entity_name"]
        for key, value in raw_data["entity_attributes"].items():
            if key in ignored_keys:
                continue

            value = str(value)

            # Remove HTML
            value = re.sub(r'<.*?>', '', value)

            # Wikipedia convert: {{convert|870|ft|m|0|abbr=on}} -> "870 ft m 0"
            value = re.sub(r'\{\{convert\|([^}]+)\}\}', lambda m: " ".join(p for p in m.group(1).split('|') if '=' not in p), value)

            # Wikipedia coord: remove completely
            value = re.sub(r'\{\{coord\|[^}]+\}\}', '', value)

            # URL template: {{URL|https://...}} -> https://...
            value = re.sub(r'\{\{URL\|([^}]+)\}\}', r'\1', value)

            # Generic Wikipedia template: {{...}} → keep parts without '='
            value = re.sub(r'\{\{([^\{\}]+)\}\}', lambda m: " ".join(p for p in m.group(1).split('|') if '=' not in p), value)

            # Wikipedia links: [[link|label]] → label, [[link]] → link
            value = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r'\2', value)
            value = re.sub(r'\[\[([^\]]+)\]\]', r'\1', value)

            # Whitespace cleanup
            value = re.sub(r'[\n\t\r]', ' ', value).strip()
            value = re.sub(r'\s+', ' ', value)

            if value and value.lower() not in ["n/a", "-", "unknown", "none"]:
                cleaned[key.replace("_", " ").lower()] = value

        return cleaned

    def summarize_data(self, image_data):
        summarization = "; ".join(f"{k} is {v}" for k, v in image_data.items())
        return summarization

    def select_topk_datas(self, image_datas, query):
        query_emb = self.semantic_model.encode(query, convert_to_tensor=True)
        corpus_emb = self.semantic_model.encode(image_datas, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, corpus_emb)[0]

        top_scores, top_indices = scores.topk(k=min(TOP_K, len(scores)))
        selected_datas = [image_datas[index] for index, score in zip(top_indices, top_scores) if score >= 0.5]
        if selected_datas == []:
            selected_datas.append(image_datas[top_indices[0]])
        return selected_datas

    def batch_generate_response(self, queries, images, message_histories=None):
        prompts = []

        for query, image in zip(queries, images):
            main_objects = self.extract_object(query)
            image = self.crop_images(image, main_objects)
            image.save("test.png")
            image_datas = self.search_pipeline(image, k=SEARCH_COUNT)

            for index, each_data in enumerate(image_datas):
                image_datas[index] = self.summarize_data(self.clean_metadata(each_data["entities"]))
            
            topk_datas = "; ".join(self.select_topk_datas(image_datas, query))

            prompt = (
                 "You are a helpful assistant which generates answer to the user question based on given information: "
                f"{topk_datas} "
                 "Answer the below question based on the given information as short and simple without any explanation, do not return full sentences " 
                 "If the given information is not enough to answer the question, just say 'I don't know' "
                f"\nUser Question: {query}"
                 "\nAnswers:"
            )

            print(prompt, end="\n\n")
            prompts.append(prompt)

        outputs = self.llm(prompts)
        print(outputs)
        answers = [output[0]["generated_text"].split("Answers:")[-1].strip() for output in outputs]

        return answers