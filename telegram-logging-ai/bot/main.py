import asyncio
import logging
import signal

from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters

import config
from .model_manager import ModelManager, get_model_manager
from .llama_client import LlamaClient
from .tts_client import TTSClient, get_tts_client
from .waifu import Waifu, get_waifu
from .journal_mode import JournalModeManager, get_journal_manager
from .journal_compiler import JournalCompiler
from .bluesky_client import BlueskyClient, get_bluesky_client
from .handlers import Handlers
from .fine_tuner import FineTuner, get_fine_tuner
from .scheduler import TaskScheduler, get_scheduler
from memory.master_repo import MasterRepository
from memory.journal_repo import JournalRepository
from memory.context import ContextBuilder, get_context_builder

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the telegram bot"""
    logger.info("Starting Telegram Logging AI with Waifu...")

    # Ensure directories exist
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.JOURNALS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize databases
    logger.info("Initializing databases...")

    master_repo = MasterRepository(
        config.MASTER_DB_PATH,
        embedding_dimension=config.EMBEDDING_DIMENSION
    )
    await master_repo.init_db()

    journal_repo = JournalRepository(config.JOURNAL_DB_PATH)
    await journal_repo.init_db()

    logger.info("Databases initialized")

    # Initialize model manager
    logger.info("Loading LLM model...")
    model_manager = get_model_manager(
        model_path=config.MAIN_MODEL_PATH,
        llama_params=config.LLAMA_PARAMS
    )

    # Load and warmup model
    await model_manager.load_model()
    await model_manager.warmup()
    logger.info("LLM model loaded and warmed up")

    # Initialize LLM client
    llm_client = LlamaClient(model_manager, config.GEN_PARAMS)

    # Initialize TTS client (optional)
    tts_client = None
    if config.TTS_ENABLED:
        try:
            tts_client = get_tts_client(
                voice_sample=config.TTS_VOICE_SAMPLE,
                use_paralinguistics=config.TTS_PARALINGUISTICS,
                output_dir=config.DATA_DIR / "audio"
            )
            logger.info("TTS client initialized")
        except Exception as e:
            logger.warning(f"TTS unavailable: {e}")

    # Initialize context builder
    context_builder = get_context_builder(
        master_repo=master_repo,
        context_hours=config.CONTEXT_HOURS,
        max_recent_messages=config.WAIFU_MAX_CONTEXT_MESSAGES,
        max_relevant_memories=config.WAIFU_MAX_RELEVANT_MEMORIES
    )

    # Initialize waifu
    waifu = get_waifu(
        llama_client=llm_client,
        context_builder=context_builder,
        master_repo=master_repo,
        personality=config.WAIFU_PERSONALITY,
        tts_client=tts_client,
        tts_enabled=config.TTS_ENABLED and config.TTS_SEND_AS_VOICE
    )
    logger.info("Waifu initialized")

    # Initialize journal components
    journal_manager = get_journal_manager(
        journal_repo=journal_repo,
        idle_warning_minutes=config.JOURNAL_IDLE_WARNING_MINUTES,
        auto_compile_hours=config.JOURNAL_AUTO_COMPILE_HOURS
    )

    journal_compiler = JournalCompiler(
        llama_client=llm_client,
        journal_repo=journal_repo,
        master_repo=master_repo,
        output_dir=config.JOURNALS_DIR
    )

    # Initialize Bluesky client
    bluesky_client = get_bluesky_client(master_repo=master_repo)

    # Check Bluesky authentication
    if await bluesky_client.is_authenticated():
        logger.info("Bluesky authenticated")
    else:
        logger.warning("Bluesky not authenticated - posting will fail")

    # Initialize fine-tuning components
    fine_tuner = None
    scheduler = None

    if config.FINE_TUNE_ENABLED:
        fine_tuner = get_fine_tuner(
            master_repo=master_repo,
            model_dir=config.MODELS_DIR,
            min_examples=config.FINE_TUNE_MIN_EXAMPLES,
            min_rating=config.FINE_TUNE_MIN_RATING
        )

        scheduler = get_scheduler(
            fine_tuner=fine_tuner,
            model_manager=model_manager,
            fine_tune_hour=config.FINE_TUNE_HOUR
        )
        scheduler.start()
        logger.info(f"Fine-tuning scheduler started (runs at {config.FINE_TUNE_HOUR}:00)")

    # Initialize handlers
    handlers = Handlers(
        waifu=waifu,
        journal_manager=journal_manager,
        journal_compiler=journal_compiler,
        bluesky_client=bluesky_client,
        llm_client=llm_client,
        allowed_users=config.TELEGRAM_ALLOWED_USERS,
        image_dir=config.IMAGE_DIR,
        tts_enabled=config.TTS_ENABLED and config.TTS_SEND_AS_VOICE
    )

    # Start journal session monitoring
    await journal_manager.start_monitoring()

    # Build Telegram application
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", handlers.cmd_start))
    app.add_handler(CommandHandler("help", handlers.cmd_help))
    app.add_handler(CommandHandler("journal", handlers.cmd_journal))
    app.add_handler(CommandHandler("bsky", handlers.cmd_bsky))
    app.add_handler(CommandHandler("voice", handlers.cmd_voice))
    app.add_handler(CommandHandler("rate", handlers.cmd_rate))
    app.add_handler(CommandHandler("status", handlers.cmd_status))

    # Register callback query handler for rating buttons
    app.add_handler(CallbackQueryHandler(handlers.callback_rate, pattern="^rate_"))

    # Register message handlers
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handlers.handle_message
    ))

    app.add_handler(MessageHandler(
        filters.PHOTO,
        handlers.handle_photo
    ))

    logger.info("Bot handlers registered")

    # Graceful shutdown handler
    async def shutdown():
        logger.info("Shutting down...")

        # Stop scheduler
        if scheduler:
            scheduler.stop()

        # Stop journal monitoring
        await journal_manager.stop_monitoring()

        # Unload models
        await model_manager.unload_model()
        if tts_client:
            tts_client.unload_model()

        logger.info("Shutdown complete")

    # Handle signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(shutdown())
        )

    # Start bot
    logger.info("Starting Telegram polling...")
    try:
        await app.run_polling(drop_pending_updates=True)
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
