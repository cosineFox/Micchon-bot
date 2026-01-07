import asyncio
import gc
from pathlib import Path
from typing import Optional
from llama_cpp import Llama
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages GGUF model loading, unloading, and hot-swapping for extreme VRAM optimization"""

    def __init__(
        self,
        model_path: Path,
        llama_params: dict,
        warmup_prompt: str = "Hello"
    ):
        """
        Initialize model manager

        Args:
            model_path: Path to GGUF model file
            llama_params: llama.cpp initialization parameters
            warmup_prompt: Prompt to use for model warmup
        """
        self.model_path = model_path
        self.llama_params = llama_params
        self.warmup_prompt = warmup_prompt
        self._model: Optional[Llama] = None
        self._vision_model: Optional[Llama] = None
        self._lock = asyncio.Lock()

    async def load_model(self) -> Llama:
        """
        Load the GGUF model into VRAM (thread-safe)

        Returns:
            Loaded Llama model instance
        """
        async with self._lock:
            if self._model is not None:
                logger.info("Model already loaded, returning existing instance")
                return self._model

            logger.info(f"Loading model from {self.model_path}")
            try:
                # Run blocking model load in thread pool
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: Llama(
                        model_path=str(self.model_path),
                        **self.llama_params
                    )
                )
                logger.info(f"Model loaded successfully - VRAM used: ~2.5GB")
                return self._model

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    async def load_vision_model(self, model_path: Path, projector_path: Path) -> Llama:
        """
        Load a separate vision model (VLM)
        """
        async with self._lock:
            if self._vision_model is not None:
                return self._vision_model

            logger.info(f"Loading vision model from {model_path} with projector {projector_path}")
            try:
                loop = asyncio.get_event_loop()
                
                def _load_vlm():
                    from llama_cpp import Llama
                    # Try to use Moondream/LLaVA handler depending on autodetection or config
                    # For generic GGUF VLMs with mmproj, we usually use Llava15ChatHandler
                    # or rely on llama-cpp-python's auto-config if chat_handler is passed
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    
                    chat_handler = Llava15ChatHandler(clip_model_path=str(projector_path))
                    
                    return Llama(
                        model_path=str(model_path),
                        chat_handler=chat_handler,
                        n_ctx=2048, # Vision context
                        n_gpu_layers=-1, # GPU offload
                        verbose=False
                    )

                self._vision_model = await loop.run_in_executor(None, _load_vlm)
                logger.info("Vision model loaded successfully")
                return self._vision_model

            except Exception as e:
                logger.error(f"Failed to load vision model: {e}")
                raise

    async def get_vision_model(self) -> Optional[Llama]:
        return self._vision_model

    async def unload_model(self):
        """
        Unload model from VRAM to free memory (thread-safe)
        """
        async with self._lock:
            if self._model is None:
                logger.info("Model not loaded, nothing to unload")
                return

            logger.info("Unloading model from VRAM")
            try:
                # Delete model and force garbage collection
                del self._model
                self._model = None
                gc.collect()

                # Try to clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("CUDA cache cleared")
                except ImportError:
                    pass

                logger.info("Model unloaded successfully")

            except Exception as e:
                logger.error(f"Error during model unload: {e}")
                raise

    async def get_model(self) -> Llama:
        """
        Get the current model instance, loading if necessary

        Returns:
            Llama model instance
        """
        if self._model is None:
            await self.load_model()
        return self._model

    async def warmup(self):
        """
        Pre-warm the model with a test inference to optimize first response time
        """
        logger.info("Warming up model...")
        model = await self.get_model()

        try:
            # Run a quick inference in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: model(
                    self.warmup_prompt,
                    max_tokens=10,
                    echo=False
                )
            )
            logger.info("Model warmup complete")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    async def reload_model(self, new_model_path: Optional[Path] = None):
        """
        Hot-swap the model (used for fine-tuning updates)

        Args:
            new_model_path: Path to new model file (if None, reloads current path)
        """
        logger.info(f"Hot-swapping model{' to ' + str(new_model_path) if new_model_path else ''}")

        # Update path if provided
        if new_model_path:
            self.model_path = new_model_path

        # Unload old model
        await self.unload_model()

        # Load new model
        await self.load_model()

        # Warmup new model
        await self.warmup()

        logger.info("Model hot-swap complete")

    async def update_model_from_path(self, model_path: Path):
        """
        Update the model to use a different model file

        Args:
            model_path: Path to the new model file
        """
        if self.model_path != model_path:
            logger.info(f"Updating model path from {self.model_path} to {model_path}")
            self.model_path = model_path

            # If model is currently loaded, reload it
            if self.is_loaded():
                await self.reload_model(model_path)

    def is_loaded(self) -> bool:
        """Check if model is currently loaded"""
        return self._model is not None

    async def get_model_info(self) -> dict:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model metadata
        """
        model = await self.get_model()

        return {
            "model_path": str(self.model_path),
            "context_size": self.llama_params.get("n_ctx", "unknown"),
            "gpu_layers": self.llama_params.get("n_gpu_layers", "unknown"),
            "loaded": self.is_loaded(),
            "n_batch": self.llama_params.get("n_batch", "unknown"),
        }


# Global singleton instance
_main_model_manager: Optional[ModelManager] = None


def get_model_manager(
    model_path: Optional[Path] = None,
    llama_params: Optional[dict] = None
) -> ModelManager:
    """
    Get or create the global model manager singleton

    Args:
        model_path: Path to model file (required on first call)
        llama_params: llama.cpp parameters (required on first call)

    Returns:
        ModelManager instance
    """
    global _main_model_manager

    if _main_model_manager is None:
        if model_path is None or llama_params is None:
            raise ValueError("model_path and llama_params required on first call")

        _main_model_manager = ModelManager(
            model_path=model_path,
            llama_params=llama_params
        )

    return _main_model_manager
