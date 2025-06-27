import json
import re

import agents.my_agent as agent
from cragmm_search.search import UnifiedSearchPipeline
from crag_image_loader import ImageLoader

image = ImageLoader("https://upload.wikimedia.org/wikipedia/commons/9/9a/202312_Dodols_sold_in_a_market_in_Sri_Lanka.jpg").get_image()

search_pipeline = UnifiedSearchPipeline(
    text_model_name=None,
    image_model_name="openai/clip-vit-large-patch14-336",
    web_hf_dataset_id=None,
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation"
)

print(agent(search_pipeline).batch_generate_response(["What is this?"], [image]))