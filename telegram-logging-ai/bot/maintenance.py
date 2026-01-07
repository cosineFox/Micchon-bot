import asyncio
import logging
import shutil
import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class MaintenanceManager:
    """Manages database backups and integrity checks"""

    def __init__(self, data_dir: Path, archive_dir: Optional[Path] = None):
        self.data_dir = data_dir
        self.archive_dir = archive_dir or (data_dir / "backups")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.dbs = ["memory.db", "journal.db"]

    async def run_maintenance(self):
        """Run full maintenance routine: backup then integrity check"""
        logger.info("Starting scheduled maintenance...")
        
        # 1. Backup
        await self.backup_databases()
        
        # 2. Integrity Check
        await self.check_integrity()
        
        # 3. Cleanup old backups (keep last 7 days)
        await self.cleanup_old_backups(days=7)
        
        logger.info("Maintenance complete.")

    async def backup_databases(self):
        """Perform hot backup of databases using VACUUM INTO"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for db_name in self.dbs:
            db_path = self.data_dir / db_name
            if not db_path.exists():
                continue
                
            backup_path = self.archive_dir / f"{db_name}_{timestamp}.bak"
            
            try:
                logger.info(f"Backing up {db_name} to {backup_path}...")
                async with aiosqlite.connect(db_path) as db:
                    await db.execute(f"VACUUM INTO '{backup_path}'")
                logger.info(f"Backup successful: {backup_path}")
            except Exception as e:
                logger.error(f"Backup failed for {db_name}: {e}")

    async def check_integrity(self):
        """Run PRAGMA integrity_check on databases"""
        for db_name in self.dbs:
            db_path = self.data_dir / db_name
            if not db_path.exists():
                continue
                
            try:
                async with aiosqlite.connect(db_path) as db:
                    async with db.execute("PRAGMA integrity_check") as cursor:
                        row = await cursor.fetchone()
                        result = row[0] if row else "Unknown"
                        
                        if result == "ok":
                            logger.info(f"Integrity check passed for {db_name}")
                        else:
                            logger.error(f"Integrity check FAILED for {db_name}: {result}")
                            # TODO: Alert user?
            except Exception as e:
                logger.error(f"Integrity check error for {db_name}: {e}")

    async def cleanup_old_backups(self, days: int = 7):
        """Remove backups older than N days"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        
        for file in self.archive_dir.glob("*.bak"):
            if file.stat().st_mtime < cutoff:
                try:
                    file.unlink()
                    logger.info(f"Deleted old backup: {file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old backup {file.name}: {e}")

def get_maintenance_manager(data_dir: Path) -> MaintenanceManager:
    return MaintenanceManager(data_dir)
