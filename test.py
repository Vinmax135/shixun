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
            "Your task is: Given an image and a user question, identify the key visual objects and their specific attributes that need to be visually located in the image in order to answer the question."
            "Only include physically visible, nameable items — such as objects, labels, signs, or container types — that are explicitly mentioned or implied in the question and that can be found visually in the image."
            "Ignore non-visual concepts like 'cost', 'price', 'calories', and ignore vague references like 'this', 'it', or abstract actions. Focus only on visually grounded items."
            "Output a comma-separated list of objects with attributes if relevant. Do not include explanations."
            "Example:"
            "Query: 'Can I throw batteries in the left bin?'"
            "Output: batteries, left bin, recycling logo"
            f"Query: {query}"
            "Answer:"
        )
inputs = vision_processor(images=image, text=prompt, return_tensors="pt").to(vision_model.device)
with torch.no_grad():
    outputs = vision_model.generate(**inputs, max_new_tokens=8)
text = vision_processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text)