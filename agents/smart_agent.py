from typing import List, Dict, Any
import torch
import re
import json
from PIL import Image
from torchvision.ops import box_convert
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from sentence_transformers import SentenceTransformer, util
from agents.base_agent import BaseAgent
from crag_web_result_fetcher import WebSearchResult

# Constants
VISION_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"  # Fits 5-8GB range
BATCH_SIZE = 1
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
SEARCH_COUNT = 10
MAX_GENERATED_TOKENS = 32

class SmartAgent(BaseAgent):
    def __init__(self, search_pipeline):
        super().__init__(search_pipeline)
        self.semantic = SentenceTransformer('all-MiniLM-L6-v2')
        self.visual_model = load_model(
            "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "../GroundingDINO/groundingdino_swint_ogc.pth"
        )
        self.vision_processor = Blip2Processor.from_pretrained(VISION_MODEL_NAME)
        self.vision_model = Blip2ForConditionalGeneration.from_pretrained(
            VISION_MODEL_NAME,
            device_map="auto",              
            offload_folder="./offload_vlm", 
            trust_remote_code=True,
            torch_dtype=torch.float16
            ).eval().cuda()

    def get_batch_size(self):
        return BATCH_SIZE

    def extract_objects_from_query(self, image: Image.Image, query: str) -> List[str]:
        prompt = (
            f"Based on the image and the question '{query}', list objects that is mentioned by the query, objects listed can be one or more, "
            "SEPARATED BY COMMAS, no explanation. For example: car, tree, person\nAnswer:"
        )
        inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
        with torch.no_grad():
            outputs = self.vision_model.generate(**inputs, max_new_tokens=8)
        text = self.vision_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(text)
        objects = [obj.strip() for obj in text.split("Answer:")[-1].split(',') if obj.strip()]
        return objects or ["item"]

    def crop_images(self, image: Image.Image, objects: List[str]) -> List[Image.Image]:
        cropped_images = []
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image.convert("RGB"), None)
        w, h = image.size

        for obj in objects:
            boxes, _, _ = predict(self.visual_model, image_tensor, obj, BOX_THRESHOLD, TEXT_THRESHOLD)
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            if len(xyxy) > 0:
                cropped_images.append(image.crop(xyxy[0]))

        return cropped_images or [image]

    def summarize_image(self, image: Image.Image) -> str:
        prompt = "Summarize the image in one sentence."
        inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
        outputs = self.vision_model.generate(**inputs, max_new_tokens=16)
        return self.vision_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    def clean_metadata(self, raw_data: List[Dict[str, Any]]) -> Dict[str, str]:
        raw_data = raw_data[0]
        cleaned = {"name": raw_data.get("entity_name", "unknown")}
        attrs = raw_data.get("entity_attributes") or {}
        ignore = {"image", "website", "url", "wikidata", "coordinates"}

        for k, v in attrs.items():
            if k in ignore: continue
            v = str(v)
            v = re.sub(r'<.*?>', '', v)
            v = re.sub(r'\{\{[^}]+\}\}', '', v)
            v = re.sub(r'\[\[[^\]]+\]\]', '', v)
            v = re.sub(r'\s+', ' ', v.strip())
            if v.lower() not in {"n/a", "none", "unknown"}:
                cleaned[k.replace("_", " ").lower()] = v
        return cleaned

    def summarize_data(self, data: Dict[str, str]) -> str:
        return ". ".join(f"{k.capitalize()}: {v}" for k, v in data.items())

    def rerank(self, query: str, summaries: List[str]) -> int:
        query_emb = self.semantic.encode(query, convert_to_tensor=True)
        scores = [util.cos_sim(query_emb, self.semantic.encode(s, convert_to_tensor=True)).item() for s in summaries]
        return scores.index(max(scores))

    def batch_generate_response(self, queries: List[str], images: List[Image.Image], message_histories=None) -> List[str]:
        responses = []
        for query, image in zip(queries, images):
            objects = self.extract_objects_from_query(image, query)

            """
            cropped_images = self.crop_images(image, objects)

            candidates = []
            for cropped in cropped_images:
                results = self.search_pipeline(cropped, k=SEARCH_COUNT)
                cleaned = [self.clean_metadata(res["entities"]) for res in results if "entities" in res and res["entities"]]
                candidates.extend(cleaned)

            image_summary = self.summarize_image(image)
            text_summaries = [self.summarize_data(c) for c in candidates]

            if not text_summaries:
                responses.append("I couldn't find enough information.")
                continue

            best_idx = self.rerank(image_summary, text_summaries)
            best_context = text_summaries[best_idx]

            # Final query for web search
            search_query = f"{query} {best_context}"
            web_results = self.search_pipeline(search_query, k=3)

            prompt = f"You are given the image summary: '{image_summary}' and the info: '{best_context}'. Answer this question: '{query}'."
            inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
            outputs = self.vision_model.generate(**inputs, max_new_tokens=MAX_GENERATED_TOKENS)
            answer = self.vision_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            responses.append(answer)
            """
        return queries
