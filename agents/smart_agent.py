from transformers import AutoModelForVision2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
import torch

from agents.base_agent import BaseAgent

class SmartAgent(BaseAgent):

    def __init__(self, search_pipeline):
        super().__init__(search_pipeline)

        # Cropping Model
        config_path = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weight_path = "../GroundingDINO/groundingdino_swint_ogc.pth"
        self.cropping_model = load_model(config_path, weight_path)

        # Vision Model
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm = AutoModelForVision2Seq.from_pretrained(
            "blip-2-flan-t5-large",
            trust_remote_code=True,
            device_map="auto",              
            offload_folder="./offload",    
            offload_state_dict=True,        
            torch_dtype=torch.float16       
        ).eval()

        # 