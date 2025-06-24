
from sentence_transformers import SentenceTransformer, util
from cragmm_search.search import UnifiedSearchPipeline
from crag_image_loader import ImageLoader

image = ImageLoader("https://upload.wikimedia.org/wikipedia/commons/b/b2/The_Beekman_tower_1_%286214362763%29.jpg").get_image()
model = SentenceTransformer("all-MiniLM-L6-v2")

search_pipeline = UnifiedSearchPipeline(
    text_model_name=None,
    image_model_name="openai/clip-vit-large-patch14-336",
    web_hf_dataset_id=None,
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation"
)
outputs = search_pipeline(image, k=5)
result = {}

for output in outputs:
    print(output)
    """
    for key, value in output.items():
        if key not in result.keys():
            result[key] = value
        
        else:
            previous = model.encode(result[key], convert_to_tensor=True)
            current = model.encode(value, convert_to_tensor=True)

            if (util.cos_sim(previous, current).item() > 0.7):
    """