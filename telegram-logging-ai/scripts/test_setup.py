#!/usr/bin/env python3
"""
Test script to verify Micchon setup.
Run: python -m scripts.test_setup
"""
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_config():
    """Test configuration loading"""
    print("=" * 50)
    print("Testing configuration...")
    try:
        import config
        print(f"  Telegram token: {'set' if config.TELEGRAM_BOT_TOKEN else 'NOT SET'}")
        print(f"  Allowed chats: {config.TELEGRAM_ALLOWED_CHAT_IDS or 'NOT SET'}")
        print(f"  Qdrant Edge: {config.QDRANT_PATH}")
        print(f"  TTS enabled: {config.TTS_ENABLED}")
        print(f"  Data dir: {config.DATA_DIR}")
        print("  [OK] Config loaded")
        return True
    except Exception as e:
        print(f"  [FAIL] Config error: {e}")
        return False


async def test_qdrant():
    """Test Qdrant Edge (embedded mode)"""
    print("=" * 50)
    print("Testing Qdrant Edge...")
    try:
        from memory.qdrant_client import get_qdrant_store
        import config

        qdrant = get_qdrant_store(
            path=config.QDRANT_PATH,
            embedding_dimension=config.EMBEDDING_DIMENSION
        )
        await qdrant.init()
        info = await qdrant.get_collection_info()
        print(f"  Storage: {config.QDRANT_PATH}")
        print(f"  Collection status: {info.get('status', 'unknown')}")
        print(f"  Vectors count: {info.get('vectors_count', 0)}")
        print("  [OK] Qdrant Edge initialized")
        return True
    except Exception as e:
        print(f"  [FAIL] Qdrant error: {e}")
        return False


async def test_embedder():
    """Test embedding model"""
    print("=" * 50)
    print("Testing embedding model...")
    try:
        from memory.embedder import get_embedder
        import config

        embedder = get_embedder(
            model_name=config.EMBED_MODEL_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            use_gpu=config.EMBEDDING_USE_GPU
        )

        # Test embedding
        test_text = "Hello, this is a test message."
        embedding = await embedder.embed(test_text)

        print(f"  Dimension: {len(embedding)}")
        print(f"  Device: {embedder.device}")
        print("  [OK] Embedder working")
        return True
    except Exception as e:
        print(f"  [FAIL] Embedder error: {e}")
        return False


async def test_llm():
    """Test LLM loading (optional - takes time)"""
    print("=" * 50)
    print("Testing LLM model...")
    try:
        import config

        if not config.MAIN_MODEL_PATH.exists():
            print(f"  [SKIP] Model not found at {config.MAIN_MODEL_PATH}")
            print("  Download from HuggingFace and place in models/")
            return None

        from bot.model_manager import get_model_manager

        manager = get_model_manager(
            model_path=config.MAIN_MODEL_PATH,
            llama_params=config.LLAMA_PARAMS
        )

        await manager.load_model()
        await manager.warmup()

        print(f"  Model: {config.MAIN_MODEL_PATH.name}")
        print("  [OK] LLM loaded and warmed up")

        await manager.unload_model()
        return True

    except Exception as e:
        print(f"  [FAIL] LLM error: {e}")
        return False


async def test_tts():
    """Test TTS (optional)"""
    print("=" * 50)
    print("Testing TTS...")
    try:
        import config

        if not config.TTS_ENABLED:
            print("  [SKIP] TTS disabled in config")
            return None

        from bot.tts_client import get_tts_client

        tts = get_tts_client(output_dir=config.DATA_DIR / "audio")

        # Don't actually generate audio in test
        print("  TTS client initialized (model loads on first use)")
        print("  [OK] TTS available")
        return True

    except ImportError as e:
        print(f"  [FAIL] Chatterbox not installed: {e}")
        print("  Install: pip install git+https://github.com/resemble-ai/chatterbox.git")
        return False
    except Exception as e:
        print(f"  [FAIL] TTS error: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("MICCHON SETUP TEST")
    print("=" * 50 + "\n")

    results = {
        "config": await test_config(),
        "qdrant": await test_qdrant(),
        "embedder": await test_embedder(),
        "tts": await test_tts(),
    }

    # LLM test is slow, ask first
    print("\n" + "=" * 50)
    print("LLM test takes ~30s. Skip? [y/N]: ", end="")

    try:
        response = input().strip().lower()
        if response != 'y':
            results["llm"] = await test_llm()
        else:
            results["llm"] = None
            print("  [SKIP] LLM test skipped")
    except EOFError:
        results["llm"] = None
        print("  [SKIP] Non-interactive mode")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for name, result in results.items():
        status = "OK" if result is True else ("FAIL" if result is False else "SKIP")
        print(f"  {name}: {status}")

    print(f"\nPassed: {passed}, Failed: {failed}, Skipped: {skipped}")

    if failed > 0:
        print("\nFix the failed tests before running the bot.")
        sys.exit(1)
    else:
        print("\nSetup looks good! Run with: python -m bot.main")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
