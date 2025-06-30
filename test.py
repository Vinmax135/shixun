from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

image = Image.open("pre.png")

VISION_MODEL_NAME = "Salesforce/blip2-opt-2.7b"

vision_processor = Blip2Processor.from_pretrained(VISION_MODEL_NAME)
vision_model = Blip2ForConditionalGeneration.from_pretrained(
VISION_MODEL_NAME,
device_map="auto",              
offload_folder="./offload_vlm", 
trust_remote_code=True,
torch_dtype=torch.float16
).eval().cuda()

prompt = f"Summarize this images."
inputs = vision_processor(images=image, text=prompt, return_tensors="pt").to(vision_model.device)
with torch.no_grad():
    outputs = vision_model.generate(**inputs, max_new_tokens=64)
text = vision_processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(text)