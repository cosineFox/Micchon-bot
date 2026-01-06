import asyncio
from datetime import datetime
from typing import Optional, Callable, Awaitable
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .fine_tuner import FineTuner
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Schedules recurring tasks like fine-tuning"""

    def __init__(
        self,
        fine_tuner: FineTuner,
        model_manager: ModelManager,
        fine_tune_hour: int = 2,
        fine_tune_minute: int = 0,
        on_fine_tune_complete: Optional[Callable[[str], Awaitable[None]]] = None
    ):
        """
        Initialize scheduler

        Args:
            fine_tuner: FineTuner instance
            model_manager: ModelManager for hot-swapping models
            fine_tune_hour: Hour to run fine-tuning (0-23)
            fine_tune_minute: Minute to run fine-tuning (0-59)
            on_fine_tune_complete: Callback when fine-tuning completes
        """
        self.fine_tuner = fine_tuner
        self.model_manager = model_manager
        self.fine_tune_hour = fine_tune_hour
        self.fine_tune_minute = fine_tune_minute
        self.on_complete = on_fine_tune_complete

        self._scheduler = AsyncIOScheduler()
        self._is_running = False

    def start(self):
        """Start the scheduler"""
        if self._is_running:
            logger.warning("Scheduler already running")
            return

        # Schedule fine-tuning job
        self._scheduler.add_job(
            self._run_fine_tuning,
            CronTrigger(
                hour=self.fine_tune_hour,
                minute=self.fine_tune_minute
            ),
            id="fine_tuning",
            name="Daily Fine-Tuning",
            replace_existing=True
        )

        # Schedule daily cleanup job
        self._scheduler.add_job(
            self._run_cleanup,
            CronTrigger(hour=3, minute=0),  # 3 AM
            id="cleanup",
            name="Daily Cleanup",
            replace_existing=True
        )

        self._scheduler.start()
        self._is_running = True

        logger.info(
            f"Scheduler started. Fine-tuning scheduled for "
            f"{self.fine_tune_hour:02d}:{self.fine_tune_minute:02d}"
        )

    def stop(self):
        """Stop the scheduler"""
        if self._is_running:
            self._scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("Scheduler stopped")

    async def _run_fine_tuning(self):
        """Run the fine-tuning job"""
        logger.info("Starting scheduled fine-tuning...")

        try:
            # Check if we should run
            should_run, reason = await self.fine_tuner.should_fine_tune()

            if not should_run:
                logger.info(f"Skipping fine-tuning: {reason}")
                return

            # Run fine-tuning
            adapter_path = await self.fine_tuner.run_fine_tuning()

            if adapter_path:
                logger.info(f"Fine-tuning complete: {adapter_path}")

                # Notify completion
                if self.on_complete:
                    await self.on_complete(f"Fine-tuning complete: {adapter_path.name}")

                # Note: Model hot-swap would happen here if using adapters
                # For GGUF models, you would need to merge LoRA and convert
                # This is left as a placeholder for future enhancement

            else:
                logger.warning("Fine-tuning did not produce an adapter")

        except Exception as e:
            logger.error(f"Fine-tuning job failed: {e}")

    async def _run_cleanup(self):
        """Run daily cleanup tasks"""
        logger.info("Running daily cleanup...")

        try:
            # Cleanup old embeddings (implemented in master_repo)
            # This is just a placeholder - actual cleanup should be called

            logger.info("Daily cleanup complete")

        except Exception as e:
            logger.error(f"Cleanup job failed: {e}")

    async def trigger_fine_tuning(self) -> tuple[bool, str]:
        """
        Manually trigger fine-tuning

        Returns:
            Tuple of (success, message)
        """
        # Check if we should run
        should_run, reason = await self.fine_tuner.should_fine_tune()

        if not should_run:
            return False, reason

        # Run in background
        asyncio.create_task(self._run_fine_tuning())

        return True, "Fine-tuning started in background"

    def get_next_run(self) -> Optional[datetime]:
        """Get the next scheduled fine-tuning time"""
        job = self._scheduler.get_job("fine_tuning")
        if job:
            return job.next_run_time
        return None

    async def get_status(self) -> dict:
        """Get scheduler status"""
        next_run = self.get_next_run()
        stats = await self.fine_tuner.get_training_stats()

        return {
            "scheduler_running": self._is_running,
            "next_fine_tune": next_run.isoformat() if next_run else None,
            "fine_tune_hour": self.fine_tune_hour,
            **stats
        }


# Global singleton
_scheduler: Optional[TaskScheduler] = None


def get_scheduler(
    fine_tuner: FineTuner,
    model_manager: ModelManager,
    fine_tune_hour: int = 2
) -> TaskScheduler:
    """Get or create global scheduler"""
    global _scheduler

    if _scheduler is None:
        _scheduler = TaskScheduler(
            fine_tuner=fine_tuner,
            model_manager=model_manager,
            fine_tune_hour=fine_tune_hour
        )

    return _scheduler
