import torch
import re
import json
from torchvision.ops import box_convert
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from sentence_transformers import SentenceTransformer, util
from json_repair import repair_json
from agents.base_agent import BaseAgent

# Constants
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 13
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
SEARCH_COUNT = 10

class MyAgent(BaseAgent):
    def __init__(self, search_pipeline):
        super().__init__(search_pipeline)
        
        # Load Visual Model
        config_path = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weight_path = "../GroundingDINO/groundingdino_swint_ogc.pth"
        self.visual_model = load_model(config_path, weight_path)

        # Load Vision Model
        self.vision_processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

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
        self.llm_extract = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=8,
            do_sample=False
        )
        self.llm_extract_description = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=128,
            do_sample=False
        )
        self.llm_generate = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=16,
            do_sample=False
        )
        self.semantic = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_batch_size(self) -> int:
        return BATCH_SIZE
    
    def crop_images(self, image, objects):                                                      # Done
        cropped_images = []                                                       
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image.convert("RGB"), None)

        for main_object in objects:
            boxes, logits, phrases = predict(
                model=self.visual_model,
                image=image_tensor,
                caption=main_object,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )

            w, h = image.size
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            if len(xyxy) > 0:
                cropped_images.append(image.crop(xyxy[0]))

        if not len(cropped_images) > 0:
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
            
        if not len(cropped_images) > 0:
            cropped_images.append(image)

        return cropped_images

    def extract_object(self, query):                                                            # Done
        prompt = f"""
            You are a helpful AI assistant specialized in understanding user queries and guiding visual search.
            Given a user query and an image, your task is to extract the main object or objects mentioned in the query that should be located in the image to answer the question.
            Only output the key object names or phrases that are visually grounded and relevant for the image search. Ignore abstract or non-visual words like "price", "cost", "calories", or vague pronouns like "this" unless they can be concretely linked to a known object.
            If the query refers vaguely (e.g., "this item") and no specific object can be extracted, respond with the most general term like "item".
            ---
            Example 1:
            User query: "What is the brand of the biscuits?"
            Output: biscuits

            Example 2:
            User query: "Can I put batteries into the left bin?"
            Output: batteries, left bin

            Example 3:
            User query: "How many calories does this item have?"
            Output: item

            Example 4:
            User query: "What type of dog is this item designed as?"
            Output: dog

            Example 5:
            User query: "How to wash it?"
            Output: item
            ---
            When generating output, keep the answer as only objects separated with ',' without any explanation
            User query: "{query}"
            Output:
            """
        output = self.llm_extract(prompt)
        responses = output[0]["generated_text"].split("Output:")[-1].strip()
        preprocessed_responses = responses.split("\n")[0].split("To")[0].split(',')
        responses_list = [response.strip() for response in preprocessed_responses]

        if responses_list[0] == '':
            responses_list[0] = "item"

        return responses_list

    def clean_metadata(self, raw_data):                                                         # Done                           
        cleaned = {}
        ignored_keys = [
            "image", "image_size", "mapframe_wikidata", "coordinates",
            "website", "url", "homepage", "official_site", "wikidata", "wikibase_item"
        ]

        raw_data = raw_data[0]
        cleaned["name"] = raw_data["entity_name"]

        if raw_data["entity_attributes"] == None:
            return {"name": raw_data["entity_name"]}

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

    def rerank(self, image_data, query):
        query_emb = self.semantic.encode(query, convert_to_tensor=True)

        reranked = []
        for info in image_data:
            candidate_text = info["name"]
            for key in info:
                if not key == "name":
                    candidate_text += " " + str(info[key])
            
            data_emb = self.semantic.encode(query, convert_to_tensor=True)
            reranked.append(util.cos_sim(query_emb, data_emb).item())

        return image_data[reranked.index(max(reranked))]
    
    def summarize_data(self, image_data):
        return ", ".join(f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in image_data.items() if v)

    def summarize_images(self, image):
        prompt = """
            <|system|>
            You are a vision expert assistant. Describe the contents of the image as clearly and completely as possible. Mention any key objects, their relationships, locations, colors, actions, and any visible text. Avoid making assumptions beyond what is clearly visible.

            <|user|>
            Describe this image in detail.
            <|image|>
        """
        inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
        with torch.no_grad():
            outputs = self.vision_model.generate(**inputs, max_new_tokens=256)
        return self.vision_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    def batch_generate_response(self, queries, images, message_histories=None):
        responses = []

        for query, image in zip(queries, images):
            main_objects = self.extract_object(query)
            cropped_images = self.crop_images(image, main_objects)

            summarization = self.summarize_images(image)
            images_datas = []
            for each_image in cropped_images:
                raw_data = self.search_pipeline(each_image, k=SEARCH_COUNT)

                cleaned_datas = []
                for each_data in raw_data:
                    cleaned_datas.append(self.clean_metadata(each_data["entities"]))

                possibly_true_data = self.rerank(cleaned_datas, summarization)
                images_datas.append(possibly_true_data)

            summarized_data = ". ".join([self.summarize_data(data) for data in images_datas])

            prompt = f"""
                <|system|>
                You are a helpful assistant. Given the image, its summary, and metadata as a support data to answer the user's question. 
                Be concise and accurate. Do not say anything other than answering the question, and no explanations for the answer. 
                If you dont know the answer, respond with: "I don't know."

                <|user|>
                Image summary: {summarization}
                Metadata: {summarized_data}
                Question: {query}
                <|image|>
                Answer:"""
            inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.vision_model.device)
            with torch.no_grad():
                outputs = self.vision_model.generate(**inputs, max_new_tokens=16)
            answer = self.vision_processor.batch_decode(outputs, skip_special_tokens=True)[0].split("Answer:")[-1].strip()
            responses.append(answer)

        return responses