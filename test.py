from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

image = Image.open("pre.png")

VISION_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"

vision_processor = Blip2Processor.from_pretrained(VISION_MODEL_NAME)
vision_model = Blip2ForConditionalGeneration.from_pretrained(
VISION_MODEL_NAME,
device_map="auto",              
offload_folder="./offload_vlm", 
trust_remote_code=True,
torch_dtype=torch.float16
).eval().cuda()

prompt = (
            f"Based on the image and the question 'How much is this scooter costs', list ONLY the key visual objects present, "
            "SEPARATED BY COMMAS, no explanation. For example: car, tree, person\nAnswer:"
        )
inputs = vision_processor(images=image, text=prompt, return_tensors="pt").to(vision_model.device)
with torch.no_grad():
    outputs = vision_model.generate(**inputs, max_new_tokens=8)
text = vision_processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text)