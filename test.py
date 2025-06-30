from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

image = Image.open("test/post2.png")

VISION_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"

vision_processor = Blip2Processor.from_pretrained(VISION_MODEL_NAME)
vision_model = Blip2ForConditionalGeneration.from_pretrained(
VISION_MODEL_NAME,
device_map="auto",              
offload_folder="./offload_vlm", 
trust_remote_code=True,
torch_dtype=torch.float16
).eval().cuda()

query = "Can i throw batteries in the left bin?"

prompt = (
            "You are a helpful AI assistant specialized in understanding user queries and guiding visual search."
            "Given a user query and an image, your task is to extract the main object or objects mentioned in the query that should be located in the image to answer the question."
            "Only output the key object names and its attributes that are visually grounded and relevant for the image search, such as left bin. Ignore abstract or non-visual words like 'price', 'cost', 'calories', or vague pronouns like 'this' unless they can be concretely linked to a known object."
            "Example: Query: 'Can I put batteries into the left bin?' Output: batteries, left bin"
            f"Query: {query}"
        )
inputs = vision_processor(images=image, text=prompt, return_tensors="pt").to(vision_model.device)
with torch.no_grad():
    outputs = vision_model.generate(**inputs, max_new_tokens=8)
text = vision_processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text)