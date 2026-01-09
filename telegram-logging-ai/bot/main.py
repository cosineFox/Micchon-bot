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
from .keyword_manager import get_keyword_manager
from memory.master_repo import MasterRepository
from memory.journal_repo import JournalRepository
from memory.context import ContextBuilder, get_context_builder
from memory.embedder import get_embedder

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def _periodic_keyword_updates(keyword_manager, master_repo):
    """Periodically update keywords based on recent memories"""
    while True:
        try:
            # Update keyword weights based on usage patterns
            await keyword_manager.adjust_weights_based_on_usage()

            # Get recent memories to extract new potential keywords
            recent_memories = await master_repo.get_recent(hours=24, limit=50)
            new_keywords = await keyword_manager.update_keywords_from_memory(recent_memories)

            if new_keywords:
                logger.info(f"Added {len(new_keywords)} new keywords from recent memories: {new_keywords}")

            # Learn from historical chats to find new patterns
            historical_keywords = await keyword_manager.learn_from_historical_chats(master_repo)
            if historical_keywords:
                logger.info(f"Learned {len(historical_keywords)} new keywords from historical chats: {historical_keywords}")

            # Sleep for 1 hour before next update
            await asyncio.sleep(3600)

        except Exception as e:
            logger.error(f"Error in periodic keyword updates: {e}")
            # Sleep for 5 minutes before retrying if there's an error
            await asyncio.sleep(300)


async def _periodic_conversation_initiation(keyword_manager, master_repo, waifu, allowed_users):
    """Periodically check if the bot should initiate a conversation"""
    last_message_times = {}  # Track last message times per user

    while True:
        try:
            # Get recent memories to determine last message times
            recent_memories = await master_repo.get_recent(hours=24, limit=100)

            # Organize memories by user
            user_memories = {}
            for memory in recent_memories:
                user_id = memory.metadata.get('user_id')
                if user_id:
                    if user_id not in user_memories:
                        user_memories[user_id] = []
                    user_memories[user_id].append(memory)

            # For each allowed user, check if we should initiate conversation
            for user_id in allowed_users:
                user_id = int(user_id)  # Ensure it's an int

                # Get user's memories
                user_recent_memories = user_memories.get(user_id, [])

                # Find last user and bot messages
                last_user_msg_time = None
                last_bot_msg_time = None

                for memory in user_recent_memories:
                    if memory.metadata.get('role') == 'user':
                        if not last_user_msg_time or memory.timestamp > last_user_msg_time:
                            last_user_msg_time = memory.timestamp
                    elif memory.metadata.get('role') == 'assistant':
                        if not last_bot_msg_time or memory.timestamp > last_bot_msg_time:
                            last_bot_msg_time = memory.timestamp

                # Check if we should initiate conversation
                should_initiate = await keyword_manager.should_initiate_conversation(
                    last_user_msg_time, last_bot_msg_time, user_id
                )

                if should_initiate:
                    logger.info(f"Initiating conversation for user {user_id}")

                    # Generate an initiative response using the LLM with full personality and context
                    try:
                        # Get context for the user
                        ctx = await waifu.context.build_context("initiating conversation", user_id)
                        context_str = waifu.context.format_context_for_prompt(ctx)
                        system_prompt = waifu._build_system_prompt(context_str)

                        # Create a prompt to initiate conversation naturally
                        llm_prompt = "The user hasn't been active for a while. Create a warm, caring check-in message that feels natural and shows you care about them. Reference something from our past conversations if possible."

                        # Generate a personalized response using the LLM with full personality and context
                        personalized_response = await waifu.llm.generate(
                            prompt=llm_prompt,
                            system=system_prompt,
                            max_tokens=100,
                            temperature=0.8
                        )

                        # Store the bot's initiated message as a memory
                        await master_repo.add_memory(
                            type="chat",
                            content=personalized_response,
                            metadata={"role": "assistant", "user_id": user_id, "initiated": True}
                        )

                        logger.info(f"Initiated conversation with: {personalized_response[:50]}...")

                    except Exception as e:
                        logger.error(f"Error generating initiative response: {e}")
                        # Fallback to a simple check-in
                        fallback_response = "Hi there! I hope you're doing well."
                        await master_repo.add_memory(
                            type="chat",
                            content=fallback_response,
                            metadata={"role": "assistant", "user_id": user_id, "initiated": True}
                        )

            # Sleep for 30 minutes before next check
            await asyncio.sleep(1800)

        except Exception as e:
            logger.error(f"Error in periodic conversation initiation: {e}")
            # Sleep for 5 minutes before retrying if there's an error
            await asyncio.sleep(300)


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

    # Load embedder first to ensure model is ready
    embedder = get_embedder(
        model_name=config.EMBED_MODEL_NAME,
        model_path=str(config.EMBEDDING_MODEL_PATH) if config.EMBEDDING_MODEL_PATH.exists() else None,
        dimension=config.EMBEDDING_DIMENSION,
        use_gpu=config.EMBEDDING_USE_GPU
    )

    master_repo = MasterRepository(
        config.MASTER_DB_PATH,
        embedding_dimension=config.EMBEDDING_DIMENSION,
        qdrant_path=config.QDRANT_PATH
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

    # Load vision model if configured
    if config.VISION_MODEL_PATH.exists() and config.VISION_PROJECTOR_PATH.exists():
        try:
            await model_manager.load_vision_model(
                config.VISION_MODEL_PATH,
                config.VISION_PROJECTOR_PATH
            )
            logger.info(f"Vision model loaded: {config.VISION_MODEL_PATH.name}")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")

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
            min_rating=config.FINE_TUNE_MIN_RATING,
            model_manager=model_manager
        )

        scheduler = get_scheduler(
            fine_tuner=fine_tuner,
            model_manager=model_manager,
            master_repo=master_repo,
            cleanup_days=config.AUTO_CLEANUP_DAYS,
            fine_tune_hour=config.FINE_TUNE_HOUR
        )
        scheduler.start()
        logger.info(f"Scheduler started (fine-tune at {config.FINE_TUNE_HOUR}:00, cleanup at 3:00)")

    # Initialize handlers
    handlers = Handlers(
        waifu=waifu,
        journal_manager=journal_manager,
        journal_compiler=journal_compiler,
        bluesky_client=bluesky_client,
        llm_client=llm_client,
        allowed_users=config.TELEGRAM_ALLOWED_CHAT_IDS,
        image_dir=config.IMAGE_DIR,
        tts_enabled=config.TTS_ENABLED and config.TTS_SEND_AS_VOICE
    )

    # Start journal session monitoring
    await journal_manager.start_monitoring()

    # Start keyword manager updates
    keyword_manager = get_keyword_manager()
    keyword_update_task = asyncio.create_task(_periodic_keyword_updates(keyword_manager, master_repo))

    # Start conversation initiation task
    conversation_init_task = asyncio.create_task(_periodic_conversation_initiation(
        keyword_manager, master_repo, waifu, config.TELEGRAM_ALLOWED_CHAT_IDS
    ))

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

        # Cancel keyword update task
        keyword_update_task.cancel()
        try:
            await keyword_update_task
        except asyncio.CancelledError:
            pass

        # Cancel conversation initiation task
        conversation_init_task.cancel()
        try:
            await conversation_init_task
        except asyncio.CancelledError:
            pass

        # Stop scheduler (handles cleanup cron job)
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
    while True:
        try:
            await app.run_polling(drop_pending_updates=True)
            # If run_polling returns, it means clean shutdown
            break
        except Exception as e:
            logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
            logger.info("Attempting to restart polling in 5 seconds...")
            await asyncio.sleep(5)
    
    await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
