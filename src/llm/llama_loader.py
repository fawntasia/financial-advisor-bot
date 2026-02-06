import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaLoader:
    def __init__(self, model_path: str = "models/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", n_ctx: int = 2048):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.model = None
        self.mock_mode = False
        
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}. Entering mock mode.")
            self.mock_mode = True
        else:
            self._load_model()

    def _detect_gpu(self) -> int:
        """
        Detects if a GPU is available and returns the number of layers to offload.
        """
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("GPU detected via torch. Offloading layers to GPU.")
                return -1  # -1 means offload all layers in llama-cpp-python
        except ImportError:
            logger.warning("Torch not installed. Could not detect GPU automatically. Defaulting to CPU.")
        
        return 0

    def _load_model(self):
        """
        Loads the Llama 3 model using llama-cpp-python.
        """
        try:
            from llama_cpp import Llama
            
            n_gpu_layers = self._detect_gpu()
            logger.info(f"Loading model from {self.model_path} with {n_gpu_layers} GPU layers.")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            logger.info("Model loaded successfully.")
        except ImportError:
            logger.error("llama-cpp-python not installed. Please install it to use the loader.")
            self.mock_mode = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.mock_mode = True

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generates text based on the provided prompt.
        """
        if self.mock_mode:
            return self._mock_generate(prompt)

        try:
            output = self.model(
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|eot_id|>"],
                echo=False
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return "I'm sorry, I encountered an error while generating a response."

    def _mock_generate(self, prompt: str) -> str:
        """
        Returns a pre-defined response for mock mode.
        """
        logger.info("Generating mock response.")
        responses = {
            "hello": "Hello! I am a Llama 3 assistant running in mock mode.",
            "stock": "In mock mode, I can't provide real-time stock advice, but Llama 3 is great for financial analysis.",
            "help": "I am here to assist you with your queries. Currently running in mock mode because the model file is missing."
        }
        
        prompt_lower = prompt.lower()
        for key in responses:
            if key in prompt_lower:
                return responses[key]
        
        return "This is a generic mock response from the Llama 3 loader. The actual model is not loaded."

if __name__ == "__main__":
    # Quick test
    loader = LlamaLoader()
    print(f"Response: {loader.generate('Hello, how are you?')}")
