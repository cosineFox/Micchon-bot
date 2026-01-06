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
        keep_versions: int = 3
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
        """
        self.repo = master_repo
        self.model_dir = model_dir
        self.min_examples = min_examples
        self.min_rating = min_rating
        self.lora_rank = lora_rank
        self.epochs = epochs
        self.keep_versions = keep_versions

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
        on_progress: Optional[callable] = None
    ) -> Optional[Path]:
        """
        Run LoRA fine-tuning

        Args:
            base_model_name: HuggingFace model to fine-tune
            on_progress: Callback for progress updates

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

            # Cleanup old versions
            await self._cleanup_old_versions()

            return adapter_path

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return None

        finally:
            self._is_training = False

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
    min_rating: int = 4
) -> FineTuner:
    """Get or create global fine-tuner"""
    global _fine_tuner

    if _fine_tuner is None:
        _fine_tuner = FineTuner(
            master_repo=master_repo,
            model_dir=model_dir,
            min_examples=min_examples,
            min_rating=min_rating
        )

    return _fine_tuner
