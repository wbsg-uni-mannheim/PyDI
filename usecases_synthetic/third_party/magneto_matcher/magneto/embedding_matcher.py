import os

import torch
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer

from .column_encoder import ColumnEncoder
from .utils.embedding_utils import compute_cosine_similarity_simple
from .utils.utils import detect_column_type, get_samples

DEFAULT_MODELS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "roberta": "sentence-transformers/all-roberta-large-v1",
    "e5": "intfloat/e5-base",
    "arctic": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2"
}

class EmbeddingMatcher:
    def __init__(self, params):
        self.params = params
        self.topk = params["topk"]
        self.embedding_threshold = params["embedding_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = params["embedding_model"]
        self.use_prompt_query = True if "arctic" in self.model_name else False

        # Load model and tokenizer
        if self.model_name in DEFAULT_MODELS:
            # Use default model directly
            model_path = DEFAULT_MODELS[self.model_name]
            self.model = SentenceTransformer(model_path, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Loaded default model '{self.model_name}' on {self.device}")
        elif "/" in self.model_name and not self.model_name.endswith((".pth", ".pt", ".bin", ".ckpt")):
            # HuggingFace model identifier (contains "/" and doesn't look like a file extension)
            # Try to load from HuggingFace
            try:
                print(f"Attempting to load HuggingFace model '{self.model_name}'...")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                print(f"Successfully loaded HuggingFace model '{self.model_name}' on {self.device}")
            except Exception as e:
                print(f"Failed to load HuggingFace model '{self.model_name}': {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to default 'mpnet' model")
                model_path = DEFAULT_MODELS["mpnet"]
                self.model = SentenceTransformer(model_path, device=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif os.path.exists(self.model_name) and os.path.isfile(self.model_name):
            # Local fine-tuned model file - check this BEFORE HuggingFace to avoid false positives
            base_key = next((key for key in DEFAULT_MODELS if key in self.model_name), "mpnet")
            if base_key not in DEFAULT_MODELS:
                print(f"Warning: No base model detected in {self.model_name}, defaulting to 'mpnet'")
                base_key = "mpnet"
            base_model_path = DEFAULT_MODELS[base_key]
            self.model = SentenceTransformer(base_model_path, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            print(f"Loaded base model '{base_key}' on {self.device}")
            print(f"Loading fine-tuned weights from {self.model_name}")
            state_dict = torch.load(self.model_name, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval().to(self.device)
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

    def _get_embeddings(self, texts, use_prompt_query=False, batch_size=32):
        """Get embeddings using Sentence Transformer's encode method"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                if use_prompt_query:
                    print("Using prompt query")
                    embeds = self.model.encode(
                        batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=self.device,
                        prompt_name="query"
                    )
                else:
                    embeds = self.model.encode(
                        batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=self.device
                    )
            embeddings.append(embeds)
        return torch.cat(embeddings)

    def get_embedding_similarity_candidates(self, source_df, target_df):
        encoder = ColumnEncoder(
            self.tokenizer,
            encoding_mode=self.params["encoding_mode"],
            sampling_mode=self.params["sampling_mode"],
            n_samples=self.params["sampling_size"],
        )

        input_col_repr_dict = {encoder.encode(source_df, col): col for col in source_df.columns}
        target_col_repr_dict = {encoder.encode(target_df, col): col for col in target_df.columns}

        cleaned_input_cols = list(input_col_repr_dict.keys())
        cleaned_target_cols = list(target_col_repr_dict.keys())

        input_embeddings = self._get_embeddings(cleaned_input_cols, self.use_prompt_query)
        target_embeddings = self._get_embeddings(cleaned_target_cols)

        top_k = min(self.topk, len(cleaned_target_cols))
        similarities, indices = compute_cosine_similarity_simple(
            input_embeddings, target_embeddings, top_k
        )

        candidates = {}
        for i, input_col in enumerate(cleaned_input_cols):
            original_input = input_col_repr_dict[input_col]
            for j in range(top_k):
                target_idx = indices[i, j]
                similarity = similarities[i, j].item()
                if similarity >= self.embedding_threshold:
                    original_target = target_col_repr_dict[cleaned_target_cols[target_idx]]
                    candidates[(original_input, original_target)] = similarity

        return candidates