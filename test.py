from PIL import Image

from cragmm_search.search import UnifiedSearchPipeline

search_api_text_model_name = None
search_api_image_model_name = "openai/clip-vit-large-patch14-336"
search_api_web_hf_dataset_id = None
search_api_image_hf_dataset_id = "crag-mm-2025/image-search-index-validation"

search_pipeline = UnifiedSearchPipeline(
    text_model_name=search_api_text_model_name,
    image_model_name=search_api_image_model_name,
    web_hf_dataset_id=search_api_web_hf_dataset_id,
    image_hf_dataset_id=search_api_image_hf_dataset_id,
)

image = Image.open("pre.png")

print(search_pipeline(image))