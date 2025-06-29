import torch
import re
import json
from torchvision.ops import box_convert
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from agents.base_agent import BaseAgent

# Constants
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 15
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
SEARCH_COUNT = 1
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
        self.llm_extract = pipeline(
            "text2text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=8,
            do_sample=False
        )
        self.llm_extract_description = pipeline(
            "text2text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=512,
            do_sample=False
        )
        self.llm_generate = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=16,
            do_sample=False
        )
    
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
            return {}

        for key, value in raw_data["entity_attributes"].items():
            if key in ignored_keys:
                continue

            value = str(value)

            if len(value) > 1000:
                cleaned.update(self.paragraph_to_dict(value))
                continue

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
    
    def paragraph_to_dict(self, text):
        prompt = f"""
            Extract structured attributes from the following product description.
            Strictly return them as a JSON object with simple field names like 'price', 'engine', 'brand', 'use_case', etc. no explanations or ideas should exist in the output.
            ---
            Example 1:
            Description:
            \"\"\"
            In 2000, Simplehuman was founded by Frank Yang, who immigrated to the United States from Taiwan in 1982 and later started the company with the idea of making a better trash can. He showed his design and received his first orders at the International Home and Housewares Show from retailers such as The Container Store and Bed Bath & Beyond. The company was originally called Canworks due to its focus on trash cans, butYang changed the name to Simplehuman in 2001 when the company began to broaden its product line into other kitchen and bath tools, under the tagline “Tools for Efficient Living”. In 2003, Simplehuman opened a UK subsidiary in Oxfordshire, England to serve the European market.
            \"\"\"
            Output:
            year_founded: 2000,
            founder: Frank Yang,
            ...
            ---
            Put answers in the output below strictly with JSON format, if there are datas about link just ignore it, keep the value as short as possible and only parse important ones.

            Description:
            \"\"\"
            {text}
            \"\"\"

            Output:
        """

        output = self.llm_extract_description(prompt)
        responses = output[0]["generated_text"]
        print(responses)
        return json.loads(responses)

    def batch_generate_response(self, queries, images, message_histories=None):
        prompts = []
        for i, (query, image) in enumerate(zip(queries, images)):
            print(f"\t\t\t\t\t {i}")
            print(f"\t\t\t\t\t {query}")
            main_objects = self.extract_object(query)
            print(f"\t\t\t\t\t {main_objects}")
            image.save(f"test/pre{i}.png")
            cropped_images = self.crop_images(image, main_objects)
            cropped_images[0].save(f"test/post{i}.png")
            images_datas = []
            for each_image in cropped_images:
                images_datas.append(self.search_pipeline(each_image, k=SEARCH_COUNT))

            for index, each_data in enumerate(images_datas):
                images_datas[index] = self.clean_metadata(each_data[0]["entities"])
                print("\n\n")
                print(images_datas[index])
                print("\n\n")

        return [query for query in queries]