import json
import re

from cragmm_search.search import UnifiedSearchPipeline
from crag_image_loader import ImageLoader

image = ImageLoader("https://upload.wikimedia.org/wikipedia/commons/b/b2/The_Beekman_tower_1_%286214362763%29.jpg").get_image()

search_pipeline = UnifiedSearchPipeline(
    text_model_name=None,
    image_model_name="openai/clip-vit-large-patch14-336",
    web_hf_dataset_id=None,
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation"
)
outputs = search_pipeline(image, k=5)

# -------- Preprocess each entity --------
image_info = []
for output in outputs:
    if not output["score"] > 0.85:
        continue
    
    for entity in output["entities"]:
        entity_result = {} # Keep information of an entity
        entity_result["entity_name"] = entity["entity_name"]

        for key, value in entity["entity_attributes"].items():
            # HTML Tags
            value = re.sub(r'<.*?>', '', value)

            # Whitespace Char
            value = re.sub(r'[\n\t\r]', '', value).strip()

            # Wikipedia Template
            value = re.sub(
                r'\{\{([^\{\}]+)\}\}',
                lambda m: ' '.join([p for p in m.group(1).split('|')[1:] if '=' not in p]),
                value
            )

            # Wikipedia Links
            value = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r' \2', value)
            value = re.sub(r'\[\[([^\]]+)\]\]', r' \1', value)

            entity_result[key] = value.strip()

        image_info.append(entity_result)