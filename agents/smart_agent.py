from typing import List, Dict, Any
import torch
import re
import json
from PIL import Image
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from sentence_transformers import SentenceTransformer, util
from agents.base_agent import BaseAgent
from crag_web_result_fetcher import WebSearchResult
from transformers import AutoProcessor, AutoModelForVision2Seq

# Constants
VISION_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
BATCH_SIZE = 2
BOX_THRESHOLD = 0.35
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
        self.vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            VISION_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

    def get_batch_size(self):
        return BATCH_SIZE

    def extract_objects_from_query(self, image: Image.Image, query: str) -> List[str]:
        system_prompt = (
            "You are a helpful AI assistant specialized in understanding user queries and guiding visual search.\n"
            "Your task: Given an image and a question, extract the visual objects or text that must be identified in the image to answer the question.\n"
            "Only output the object names or visual labels relevant for the query. No explanations.\n"
            "If vague (e.g., 'this'), return 'item'.\n"
            "Format: comma-separated."
        )
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\nQuery: {query}\n<|image|>"
        inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
        with torch.no_grad():
            outputs = self.vision_model.generate(**inputs, max_new_tokens=16)
        text = self.vision_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        objects = [obj.strip() for obj in re.split(r"[\,\n]", text) if obj.strip()]
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

        if not cropped_images:
            boxes, logits, phrases = predict(
                model=self.visual_model,
                image=image_tensor,
                caption="all objects",
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            if len(xyxy) > 0:
                areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in xyxy]
                largest_idx = areas.index(max(areas))
                cropped_images.append(image.crop(xyxy[largest_idx]))

        return cropped_images or [image]

    def summarize_image(self, image: Image.Image) -> str:
        prompt = "<|system|>\nYou are a helpful assistant. Describe the image in detail.\n<|user|>\n<|image|>"
        inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
        with torch.no_grad():
            outputs = self.vision_model.generate(**inputs, max_new_tokens=128)
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
            cropped_images = self.crop_images(image, objects)

            candidates = []
            for cropped in cropped_images:
                results = self.search_pipeline(cropped, k=SEARCH_COUNT)
                cleaned = [self.clean_metadata(res["entities"]) for res in results if "entities" in res and res["entities"]]
                candidates.extend(cleaned)

            image_summary = self.summarize_image(cropped_images[0])
            text_summaries = [self.summarize_data(c) for c in candidates]

            if not text_summaries:
                responses.append("I couldn't find enough information.")
                continue

            best_idx = self.rerank(image_summary, text_summaries)
            best_context = text_summaries[best_idx]

            prompt = (
                "<|system|>\nYou are a helpful assistant answering based on visual content and extra information.\n"
                f"<|user|>\nImage summary: {image_summary}\nMetadata: {best_context}\nQuestion: {query}\n<|image|>"
            )
            inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
            with torch.no_grad():
                outputs = self.vision_model.generate(**inputs, max_new_tokens=MAX_GENERATED_TOKENS)
            answer = self.vision_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            responses.append(answer)

        return responses
