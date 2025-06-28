import torch
import re
import numpy as np
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForZeroShotObjectDetection
from sentence_transformers import SentenceTransformer, util
from segment_anything import sam_model_registry, SamPredictor

from agents.base_agent import BaseAgent

# Constants
EXTRACTOR_MODEL_NAME = "en_core_web_sm"
VISUAL_MODEL_NAME = "IDEA-Research/grounding-dino-base"
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SAM_CHECKPOINT = "./segment-anything/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
BATCH_SIZE = 1
TEXT_THRESHOLD = 0.5
SEARCH_COUNT = 10
TOP_K = 3

class MyAgent(BaseAgent):
    def __init__(self, search_pipeline):
        super().__init__(search_pipeline)
        
        # Load Visual Model
        self.visual_processor = AutoProcessor.from_pretrained(VISUAL_MODEL_NAME)
        self.visual_model = AutoModelForZeroShotObjectDetection.from_pretrained(VISUAL_MODEL_NAME).to("cuda")

        # SAM Model
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to("cuda")
        self.sam_predictor = SamPredictor(sam)

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
        inputs = self.visual_processor(images=image, text="scooter.", return_tensors="pt").to(self.visual_model.device)
        with torch.no_grad():
            outputs = self.visual_model(**inputs)

        logits = outputs.logits
        boxes = outputs.pred_boxes
        scores = torch.sigmoid(logits[0])
        max_scores, _ = scores.max(dim=1)
        keep = max_scores > TEXT_THRESHOLD

        if not keep.any():
            print("❗No object matched the query. Returning full image.")
            return image

        kept_boxes = boxes[0][keep].cpu().numpy()
        # Choose the largest box by area
        areas = (kept_boxes[:,2] - kept_boxes[:,0]) * (kept_boxes[:,3] - kept_boxes[:,1])
        best_idx = int(np.argmax(areas))
        box_norm = kept_boxes[best_idx]

        # Convert normalized box to pixel coords
        w, h = image.size
        x1, y1, x2, y2 = [int(box_norm[i] * (w if i%2==0 else h)) for i in range(4)]
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))

        # Step 2: SAM for fine mask
        image_np = np.array(image)
        self.sam_predictor.set_image(image_np)
        masks, _, _ = self.sam_predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            box_format="xyxy",
            multimask_output=False
        )

        mask = masks[0]
        masked_img = image_np.copy()
        masked_img[~mask] = 0

        return Image.fromarray(masked_img)

    def extract_object(self, query):
        prompt = (
            "Extract all real-world objects (like physical things) mentioned in this query. "
            "Return them as a comma-separated list with no explanation, Do not return full sentences.\n"
            f"Query: {query}\nObjects:"
        )

        output = self.llm(prompt)[0]["generated_text"]
        result = output.split("Objects:")[-1].strip()

        # Optionally format result as 'x. y. z.'
        result = result.replace(",", ".").replace("..", ".").strip()
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