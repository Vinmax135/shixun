import torch
import re
from torchvision.ops import box_convert
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from agents.base_agent import BaseAgent

# Constants
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 1
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
        self.llm = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=16,
            do_sample=False
        )
    
    def get_batch_size(self) -> int:
        return BATCH_SIZE
    
    def crop_images(self, image, objects):
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
            cropped_images.append(image)

        return cropped_images

    def extract_object(self, query):  
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
        output = self.llm(prompt)
        responses = output[0]["generated_text"].split("Output:")[-1].strip()
        preprocessed_responses = responses.split("\n")[0].split("To")[0].split(',')
        responses_list = [response.strip() for response in preprocessed_responses]

        if responses_list[0] == '':
            responses_list[0] = "item"

        return responses_list

    def clean_metadata(self, raw_data):                                                   
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
        summarization = ", ".join(f"{k} is {v}" for k, v in image_data.items())
        return summarization

    def batch_generate_response(self, queries, images, message_histories=None):
        prompts = []
        for i, (query, image) in enumerate(zip(queries, images)):
            print(f"\t\t\t\t\t {i}")
            print(f"\t\t\t\t\t {query}")
            main_objects = self.extract_object(query)
            print(f"\t\t\t\t\t {main_objects}")
            image.save(f"test/pre{i}.png")
            cropped_images = self.crop_images(image, main_objects)
            image.save(f"test/post{i}.png")
            images_datas = []
            for each_image in cropped_images:
                images_datas.append(self.search_pipeline(each_image, k=SEARCH_COUNT))

            for index, each_data in enumerate(images_datas):
                images_datas[index] = self.summarize_data(self.clean_metadata(each_data[0]["entities"]))

            information = "\n\n".join(images_datas)

            prompt = (
                 "You are a helpful assistant which generates answer to the user question based on given information: \n"
                f"{information} "
                 "Answer the below question based on the given information as short and simple without any explanation, do not return full sentences " 
                 "If the given information is not enough to answer the question, just say 'I don't know' "
                f"\nUser Question: {query}"
                 "\nAnswers:"
            )
            prompts.append(prompt)

        outputs = self.llm(prompts)
        answers = [output[0]["generated_text"].split("Answers:")[-1].strip().split("\n")[0] for output in outputs]

        return answers