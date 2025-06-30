from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

image = Image.open("test/post1.png")

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

prompt = f"""
            You are a helpful assistant, describe this image, make sure those are the key points of the given image.
        """
inputs = vision_processor(images=image, text=prompt, return_tensors="pt").to(vision_model.device)
with torch.no_grad():
    outputs = vision_model.generate(**inputs, max_new_tokens=128)
text = vision_processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text)