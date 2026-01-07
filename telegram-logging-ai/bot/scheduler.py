import asyncio
from datetime import datetime
from typing import Optional, Callable, Awaitable
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .fine_tuner import FineTuner
from .model_manager import ModelManager
from .maintenance import get_maintenance_manager
from memory.master_repo import MasterRepository
import config

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Schedules recurring tasks like fine-tuning and cleanup"""

    def __init__(
        self,
        fine_tuner: FineTuner,
        model_manager: ModelManager,
        master_repo: MasterRepository,
        cleanup_days: int = 90,
        fine_tune_hour: int = 2,
        fine_tune_minute: int = 0,
        on_fine_tune_complete: Optional[Callable[[str], Awaitable[None]]] = None
    ):
        """
        Initialize scheduler

        Args:
            fine_tuner: FineTuner instance
            model_manager: ModelManager for hot-swapping models
            master_repo: MasterRepository for cleanup operations
            cleanup_days: Days to keep memories (older are deleted)
            fine_tune_hour: Hour to run fine-tuning (0-23)
            fine_tune_minute: Minute to run fine-tuning (0-59)
            on_fine_tune_complete: Callback when fine-tuning completes
        """
        self.fine_tuner = fine_tuner
        self.model_manager = model_manager
        self.master_repo = master_repo
        self.cleanup_days = cleanup_days
        self.fine_tune_hour = fine_tune_hour
        self.fine_tune_minute = fine_tune_minute
        self.on_complete = on_fine_tune_complete
        self.maintenance = get_maintenance_manager(config.DATA_DIR)

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

        # Schedule maintenance job (Backup + Integrity Check) at 2:30 AM
        self._scheduler.add_job(
            self.maintenance.run_maintenance,
            CronTrigger(hour=2, minute=30),
            id="maintenance",
            name="Database Maintenance",
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

            # Run fine-tuning with GGUF merging enabled
            adapter_path = await self.fine_tuner.run_fine_tuning(merge_to_gguf=True)

            if adapter_path:
                logger.info(f"Fine-tuning complete: {adapter_path}")

                # Notify completion
                if self.on_complete:
                    await self.on_complete(f"Fine-tuning complete: {adapter_path.name}")

                # If we have a merged model and a model manager, update the model
                if adapter_path.suffix == '.gguf' and self.model_manager:
                    logger.info(f"Updating model with new GGUF: {adapter_path}")
                    try:
                        await self.model_manager.update_model_from_path(adapter_path)
                        logger.info("Model updated successfully")
                    except Exception as e:
                        logger.error(f"Failed to update model: {e}")

            else:
                logger.warning("Fine-tuning did not produce an adapter")

        except Exception as e:
            logger.error(f"Fine-tuning job failed: {e}")

    async def _run_cleanup(self):
        """Run daily cleanup tasks - deletes old memories and embeddings"""
        logger.info(f"Running daily cleanup (keeping last {self.cleanup_days} days)...")

        try:
            await self.master_repo.cleanup_old_memories(days=self.cleanup_days)
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
    master_repo: MasterRepository,
    cleanup_days: int = 90,
    fine_tune_hour: int = 2
) -> TaskScheduler:
    """Get or create global scheduler"""
    global _scheduler

    if _scheduler is None:
        _scheduler = TaskScheduler(
            fine_tuner=fine_tuner,
            model_manager=model_manager,
            master_repo=master_repo,
            cleanup_days=cleanup_days,
            fine_tune_hour=fine_tune_hour
        )

    return _scheduler
