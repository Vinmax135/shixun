import torch
import numpy as np
import cv2
import re
import json
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from groundingdino.util.inference import load_model, predict
from groundingdino.util.inference import annotate

from agents.base_agent import BaseAgent

# Constants
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 1
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
SEARCH_COUNT = 10
TOP_K = 3

class MyAgent(BaseAgent):
    def __init__(self, search_pipeline):
        super().__init__(search_pipeline)
        
        # Load Visual Model
        CONFIG_PATH = "groundingdino_swinb_cfg.py"
        WEIGHT_PATH = "groundingdino_swinb.pth"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        visual_model = load_model(CONFIG_PATH, WEIGHT_PATH)
        visual_model.to(DEVICE)

        # Load LLM
        offload_folder = "./offload_myagent"
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            offload_folder=offload_folder,
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
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
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        boxes, phrases = predict(
            model=self.visual_model,
            image=image_cv,
            caption=query,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=self.DEVICE
        )

        if len(boxes) == 0:
            print("❗No object matched the query. Returning full image.")
            return image
        
        h, w, _ = image_cv.shape
        x1, y1, x2, y2 = boxes[0]
        x1 = max(int(x1 * w), 0)
        y1 = max(int(y1 * h), 0)
        x2 = min(int(x2 * w), w)
        y2 = min(int(y2 * h), h)

        return image.crop((x1, y1, x2, y2))

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
        selected_datas = [image_datas[index] for index, score in zip(top_indices, top_scores) if score.item() >= 0.5]
        return selected_datas

    def batch_generate_response(self, queries, images, message_histories=None):
        prompts = []

        for query, image in zip(queries, images):
            image = self.crop_images(image, query)
            image_datas = self.search_pipeline(image, k=SEARCH_COUNT)
            
            for index, each_data in enumerate(image_datas):
                image_datas[index] = self.summarize_data(self.clean_metadata(each_data["entities"]))
            
            topk_datas = "; ".join(self.select_topk_datas(image_datas, query))

            prompt = (
                 "You are a helpful assistant which generates answer to the user question based on given information: "
                f"{topk_datas}"
                 "Answer the below question based on the given information as short and simple as possible, " 
                 "If the given information is not enough to answer the question, say 'I don't know'"
                f"User Question: {query}"
                 "Answers:"
            )

            print(prompt, end="\n\n")
            prompts.append(prompt)

        outputs = self.llm(prompts)[0]["generated_text"]
        answers = [output.split("Answers:")[-1].strip() for output in outputs]

        return answers