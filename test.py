import json
import re

from PIL import Image
import agents.my_agent as agent
from cragmm_search.search import UnifiedSearchPipeline
from crag_image_loader import ImageLoader

image = Image.open("./image.png")

search_pipeline = UnifiedSearchPipeline(
    text_model_name=None,
    image_model_name="openai/clip-vit-large-patch14-336",
    web_hf_dataset_id=None,
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation"
)

test = agent.MyAgent(search_pipeline)
print(test.batch_generate_response(["What is the cost of the bike?"], [image]))