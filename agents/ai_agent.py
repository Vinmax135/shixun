import re
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

from base_agent import BaseAgent

# Constants
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 1
TOP_K = 3  # Top-k entities for context building


class AIAgent(BaseAgent):
    def __init__(self, search_pipeline):
        super().__init__(search_pipeline)
        
        # Load LLM
        offload_folder = "./offload_myagent"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            offload_folder=offload_folder,
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        self.llm = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=16,
            do_sample=False
        )
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Initializing AIAgent")

    def get_batch_size(self) -> int:
        return BATCH_SIZE

    def clean_metadata(self, raw):
        cleaned = {}
        ignored_keys = [
            "image", "image_size", "mapframe_wikidata", "coordinates",
            "website", "url", "homepage", "official_site", "wikidata", "wikibase_item"
        ]

        for key, value in raw.items():
            if key in ignored_keys:
                continue

            value = str(value)

            # Remove HTML
            value = re.sub(r'<.*?>', '', value)

            # Wikipedia convert: {{convert|870|ft|m|0|abbr=on}} -> "870 ft m 0"
            value = re.sub(r'\{\{convert\|([^}]+)\}\}', lambda m: " ".join(p for p in m.group(1).split('|') if '=' not in p), value)

            # Wikipedia coord: remove completely
            value = re.sub(r'\{\{coord\|[^}]+\}\}', '', value)

            # URL template: {{URL|https://...}} -> https://...
            value = re.sub(r'\{\{URL\|([^}]+)\}\}', r'\1', value)

            # Generic Wikipedia template: {{...}} → keep parts without '='
            value = re.sub(r'\{\{([^\{\}]+)\}\}', lambda m: " ".join(p for p in m.group(1).split('|') if '=' not in p), value)

            # Wikipedia links: [[link|label]] → label, [[link]] → link
            value = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r'\2', value)
            value = re.sub(r'\[\[([^\]]+)\]\]', r'\1', value)

            # Whitespace cleanup
            value = re.sub(r'[\n\t\r]', ' ', value).strip()
            value = re.sub(r'\s+', ' ', value)

            if value and value.lower() not in ["n/a", "-", "unknown", "none"]:
                cleaned[key.replace("_", " ").lower()] = value

        return cleaned

    def merge_cleaned_entities(self, entities: list[dict]) -> dict:
        merged = {}
        for ent in entities:
            for k, v in ent.items():
                if k not in merged:
                    merged[k] = v
                elif v not in merged[k]:
                    merged[k] += "; " + v
        return merged

    def select_topk_entities(self, entities, query: str, topk: int = TOP_K):
        texts = []
        clean_entities = []

        for entity in entities:
            if entity.get("entity_attributes"):
                cleaned = self.clean_metadata(entity["entity_attributes"])
                clean_entities.append(cleaned)

                text = f"{entity.get('entity_name', '')}. " + ". ".join(f"{k} is {v}" for k, v in cleaned.items())
                texts.append(text)

        if not texts:
            return {}

        query_emb = self.semantic_model.encode(query, convert_to_tensor=True)
        corpus_emb = self.semantic_model.encode(texts, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, corpus_emb)[0]

        # Top-k selection with threshold filtering
        top_scores, top_indices = scores.topk(k=min(topk, len(scores)))
        filtered = [(i, s.item()) for i, s in zip(top_indices, top_scores) if s.item() >= 0.5]

        print("\nTop-k entities and scores:")
        for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), start=1):
            preview = texts[idx][:80].replace("\n", " ")
            print(f"{rank}. Score: {score:.3f} | Entity preview: {preview}...")

        if filtered:
            selected = [clean_entities[i] for i, _ in filtered]
            return self.merge_cleaned_entities(selected)
        else:
            print("⚠️ No entity passed threshold; falling back to top-1 entity")
            return clean_entities[top_indices[0]]

    def build_prompt(self, context, query):
        context_str = json.dumps(context, ensure_ascii=False)
        return (
            "You are a fact-based assistant.\n"
            "Use only the given information to answer the user question.\n"
            "If the answer is not clearly present, say: I don't know.\n"
            f"\nInformation:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        )

    def process(self, images, queries):
        answers = []
        for img, query in zip(images, queries):
            results = self.search(img, k=10)
            entities = [r["entities"][0] for r in results if r.get("entities")]
            best_info = self.select_topk_entities(entities, query)

            prompt = self.build_prompt(best_info, query)
            output = self.llm(prompt)[0]["generated_text"]
            answer = output.split("Answer:")[-1].strip()
            answers.append(answer)
        return answers

    def batch_generate_response(self, queries, images, message_histories=None):
        return self.process(images, queries)