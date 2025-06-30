from cragmm_search.search import UnifiedSearchPipeline
from crag_image_loader import ImageLoader

search_api_text_model_name = "Qwen/Qwen1.5-0.5B-Chat"
search_api_image_model_name = "openai/clip-vit-large-patch14-336"
search_api_web_hf_dataset_id = "crag-mm-2025/web-search-index-validation"
search_api_image_hf_dataset_id = "crag-mm-2025/image-search-index-validation"

search_pipeline = UnifiedSearchPipeline(
    text_model_name=search_api_text_model_name,
    image_model_name=search_api_image_model_name,
    web_hf_dataset_id=search_api_web_hf_dataset_id,
    image_hf_dataset_id=search_api_image_hf_dataset_id,
)

query = "vespa gts super 300?"

print(search_pipeline(query))