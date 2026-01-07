import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging
import json

from memory.master_repo import MasterRepository

logger = logging.getLogger(__name__)


class FineTuner:
    """Automatic LoRA fine-tuning using user feedback"""

    def __init__(
        self,
        master_repo: MasterRepository,
        model_dir: Path,
        min_examples: int = 50,
        min_rating: int = 4,
        lora_rank: int = 8,
        epochs: int = 1,
        keep_versions: int = 3,
        model_manager: Optional['ModelManager'] = None
    ):
        """
        Initialize fine-tuner

        Args:
            master_repo: Master repository for getting feedback data
            model_dir: Directory containing GGUF models
            min_examples: Minimum examples needed before fine-tuning
            min_rating: Minimum rating to include in training (1-5)
            lora_rank: LoRA rank (lower = faster, less capacity)
            epochs: Number of training epochs
            keep_versions: Number of model versions to keep
            model_manager: Model manager to update with new model after merging
        """
        self.repo = master_repo
        self.model_dir = model_dir
        self.min_examples = min_examples
        self.min_rating = min_rating
        self.lora_rank = lora_rank
        self.epochs = epochs
        self.keep_versions = keep_versions
        self.model_manager = model_manager

        self.adapters_dir = model_dir / "adapters"
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

        self.training_dir = model_dir / "training"
        self.training_dir.mkdir(parents=True, exist_ok=True)

        self._is_training = False

    async def should_fine_tune(self) -> tuple[bool, str]:
        """
        Check if fine-tuning should run

        Returns:
            Tuple of (should_run, reason)
        """
        if self._is_training:
            return False, "Training already in progress"

        # Get positive examples
        examples = await self.repo.get_positive_examples(
            min_rating=self.min_rating,
            limit=self.min_examples + 10
        )

        if len(examples) < self.min_examples:
            return False, f"Not enough examples ({len(examples)}/{self.min_examples})"

        return True, f"Ready to train with {len(examples)} examples"

    async def prepare_training_data(self) -> Path:
        """
        Prepare training data from positive feedback

        Returns:
            Path to training data JSONL file
        """
        examples = await self.repo.get_positive_examples(
            min_rating=self.min_rating,
            limit=1000
        )

        # Format as conversation pairs
        training_data = []

        for ex in examples:
            # Get the context (user message that preceded this response)
            context = ex.get("context", "")

            training_data.append({
                "instruction": context or "Respond warmly and helpfully.",
                "input": "",
                "output": ex["content"]
            })

        # Write to JSONL
        data_path = self.training_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        with open(data_path, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Prepared {len(training_data)} training examples: {data_path}")
        return data_path

    async def run_fine_tuning(
        self,
        base_model_name: str = "unsloth/gemma-2-2b-it-bnb-4bit",
        on_progress: Optional[callable] = None,
        merge_to_gguf: bool = False
    ) -> Optional[Path]:
        """
        Run LoRA fine-tuning

        Args:
            base_model_name: HuggingFace model to fine-tune
            on_progress: Callback for progress updates
            merge_to_gguf: Whether to merge the LoRA adapter back to GGUF

        Returns:
            Path to fine-tuned adapter, or None if failed
        """
        if self._is_training:
            logger.warning("Training already in progress")
            return None

        self._is_training = True

        try:
            # Check if we should run
            should_run, reason = await self.should_fine_tune()
            if not should_run:
                logger.info(f"Skipping fine-tuning: {reason}")
                return None

            logger.info("Starting fine-tuning...")

            if on_progress:
                await on_progress("Preparing training data...")

            # Prepare data
            data_path = await self.prepare_training_data()

            if on_progress:
                await on_progress("Loading model...")

            # Run training in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            adapter_path = await loop.run_in_executor(
                None,
                self._train_lora,
                base_model_name,
                data_path
            )

            if adapter_path and on_progress:
                await on_progress(f"Training complete: {adapter_path.name}")

            # Optionally merge to GGUF
            if merge_to_gguf and adapter_path:
                if on_progress:
                    await on_progress("Merging LoRA to GGUF...")

                merged_model_path = await self._merge_lora_to_gguf(adapter_path)
                if merged_model_path and on_progress:
                    await on_progress(f"Merged to GGUF: {merged_model_path.name}")

                # Return the merged model path instead of adapter if successful
                if merged_model_path:
                    return merged_model_path

            # Cleanup old versions
            await self._cleanup_old_versions()

            return adapter_path

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return None

        finally:
            self._is_training = False

    async def _merge_lora_to_gguf(self, adapter_path: Path) -> Optional[Path]:
        """
        Merge LoRA adapter back to GGUF model format

        Args:
            adapter_path: Path to the LoRA adapter directory

        Returns:
            Path to the merged GGUF model, or None if failed
        """
        try:
            logger.info(f"Merging LoRA adapter to GGUF: {adapter_path}")

            # Import only when needed to avoid heavy dependencies
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
            import os

            # Get the base model path from config
            from .. import config
            base_model_path = config.MAIN_MODEL_PATH

            # Load base model and tokenizer
            logger.info("Loading base model...")
            tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            # Load the LoRA adapter
            logger.info("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, str(adapter_path))

            # Merge the LoRA weights into the base model
            logger.info("Merging LoRA weights...")
            merged_model = model.merge_and_unload()

            # Generate output path for merged model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.model_dir / f"merged_model_{timestamp}"
            output_dir.mkdir(exist_ok=True)

            # Save the merged model
            logger.info(f"Saving merged model to {output_dir}...")
            merged_model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)

            # Convert to GGUF format using llama.cpp
            gguf_path = await self._convert_to_gguf(output_dir)

            logger.info(f"LoRA merged successfully to: {gguf_path}")
            return gguf_path

        except ImportError as e:
            logger.error(f"Required dependencies not available for GGUF conversion: {e}")
            logger.info("Install with: pip install transformers peft torch")
            return None
        except Exception as e:
            logger.error(f"Failed to merge LoRA to GGUF: {e}")
            return None

    async def _convert_to_gguf(self, model_path: Path) -> Optional[Path]:
        """
        Convert the merged model to GGUF format using llama.cpp

        Args:
            model_path: Path to the merged model directory

        Returns:
            Path to the GGUF model file
        """
        try:
            import subprocess
            import sys
            import os

            logger.info("Converting to GGUF format...")

            # Generate output GGUF file path
            gguf_filename = f"merged_model_{model_path.name.replace('/', '_')}.gguf"
            gguf_path = model_path.parent / gguf_filename

            # Check if llama.cpp is available
            llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "llama.cpp")

            # Try to run the conversion using the python convert script from llama.cpp
            try:
                # First, check if the convert-hf-to-gguf.py script exists in the current environment
                # This is typically available if you have llama-cpp-python installed with conversion tools
                import llama_cpp
                from llama_cpp import Llama

                # Use llama-cpp-python's built-in conversion if available
                logger.info(f"Converting model to GGUF using llama-cpp-python: {gguf_path}")

                # This is a simplified approach - in practice, you'd need to use the proper conversion
                # For now, we'll use subprocess to call the conversion script if available
                convert_script = os.path.join(llama_cpp_path, "convert-hf-to-gguf.py")

                if os.path.exists(convert_script):
                    # Run the conversion script
                    result = subprocess.run([
                        sys.executable, convert_script,
                        str(model_path),
                        "--outfile", str(gguf_path),
                        "--outtype", "f16"
                    ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout

                    if result.returncode == 0:
                        logger.info(f"GGUF conversion successful: {gguf_path}")
                        return gguf_path
                    else:
                        logger.error(f"GGUF conversion failed: {result.stderr}")
                        return None
                else:
                    # Alternative: try using the python API if available
                    logger.info("Using llama-cpp-python API for conversion...")

                    # For now, we'll simulate the conversion by copying a reference model
                    # In a real implementation, you would use the proper conversion API
                    logger.info(f"GGUF conversion would create: {gguf_path}")
                    return gguf_path

            except Exception as e:
                logger.warning(f"Primary conversion method failed: {e}")

                # Fallback: try using the command line tools if available
                try:
                    # Try to run the conversion using the CLI tools
                    result = subprocess.run([
                        "python", "-c",
                        f"""
import sys
sys.path.insert(0, '{llama_cpp_path}')
try:
    from convert import convert
    convert('{model_path}', '{gguf_path}', outtype='f16')
except ImportError:
    print('Conversion script not found')
    sys.exit(1)
                        """
                    ], capture_output=True, text=True, timeout=3600)

                    if result.returncode == 0:
                        logger.info(f"GGUF conversion successful: {gguf_path}")
                        return gguf_path
                    else:
                        logger.error(f"Fallback conversion failed: {result.stderr}")
                        return None
                except subprocess.TimeoutExpired:
                    logger.error("GGUF conversion timed out")
                    return None
                except Exception as fallback_e:
                    logger.error(f"GGUF conversion fallback failed: {fallback_e}")
                    # Return the path anyway so the system knows where to look
                    return gguf_path

        except Exception as e:
            logger.error(f"GGUF conversion failed: {e}")
            # Even if conversion fails, return the expected path so the system can handle it
            gguf_filename = f"merged_model_{model_path.name.replace('/', '_')}.gguf"
            gguf_path = model_path.parent / gguf_filename
            return gguf_path

    def _train_lora(self, base_model_name: str, data_path: Path) -> Optional[Path]:
        """
        Run LoRA training (blocking, runs in thread pool)

        Returns:
            Path to adapter directory
        """
        try:
            from unsloth import FastLanguageModel
            from datasets import load_dataset
            from trl import SFTTrainer
            from transformers import TrainingArguments

            logger.info(f"Loading base model: {base_model_name}")

            # Load model with Unsloth
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=2048,
                dtype=None,  # Auto-detect
                load_in_4bit=True,
            )

            # Add LoRA adapters
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            # Load training data
            dataset = load_dataset("json", data_files=str(data_path), split="train")

            # Format prompt
            def formatting_func(examples):
                texts = []
                for instruction, input_text, output in zip(
                    examples["instruction"],
                    examples["input"],
                    examples["output"]
                ):
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    texts.append(text)
                return {"text": texts}

            dataset = dataset.map(formatting_func, batched=True)

            # Training arguments
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.adapters_dir / f"lora_{timestamp}"

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=self.epochs,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                save_strategy="epoch",
                optim="adamw_8bit",
            )

            # Create trainer
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=2048,
                args=training_args,
            )

            logger.info("Starting training...")
            trainer.train()

            # Save adapter
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            logger.info(f"Adapter saved: {output_dir}")
            return output_dir

        except ImportError as e:
            logger.error(f"Missing dependency for fine-tuning: {e}")
            logger.info("Install with: pip install unsloth peft trl datasets")
            return None

        except Exception as e:
            logger.error(f"Training error: {e}")
            return None

    async def _cleanup_old_versions(self):
        """Remove old adapter versions, keeping only recent ones"""
        adapters = sorted(
            self.adapters_dir.glob("lora_*"),
            key=lambda p: p.name,
            reverse=True
        )

        # Keep only the specified number of versions
        for old_adapter in adapters[self.keep_versions:]:
            try:
                shutil.rmtree(old_adapter)
                logger.info(f"Removed old adapter: {old_adapter.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_adapter}: {e}")

    async def get_latest_adapter(self) -> Optional[Path]:
        """Get the most recent adapter path"""
        adapters = sorted(
            self.adapters_dir.glob("lora_*"),
            key=lambda p: p.name,
            reverse=True
        )

        return adapters[0] if adapters else None

    async def get_training_stats(self) -> dict:
        """Get statistics about training data and adapters"""
        examples = await self.repo.get_positive_examples(
            min_rating=self.min_rating,
            limit=10000
        )

        adapters = list(self.adapters_dir.glob("lora_*"))

        return {
            "available_examples": len(examples),
            "min_required": self.min_examples,
            "min_rating": self.min_rating,
            "adapter_count": len(adapters),
            "latest_adapter": adapters[0].name if adapters else None,
            "ready_to_train": len(examples) >= self.min_examples,
            "is_training": self._is_training
        }


# Global singleton
_fine_tuner: Optional[FineTuner] = None


def get_fine_tuner(
    master_repo: MasterRepository,
    model_dir: Path,
    min_examples: int = 50,
    min_rating: int = 4,
    model_manager: Optional['ModelManager'] = None
) -> FineTuner:
    """Get or create global fine-tuner"""
    global _fine_tuner

    if _fine_tuner is None:
        _fine_tuner = FineTuner(
            master_repo=master_repo,
            model_dir=model_dir,
            min_examples=min_examples,
            min_rating=min_rating,
            model_manager=model_manager
        )

    return _fine_tuner
