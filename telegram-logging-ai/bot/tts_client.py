import asyncio
import tempfile
from pathlib import Path
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)

# Lazy import to avoid loading heavy dependencies on import
_chatterbox_model = None


class TTSClient:
    """Chatterbox-Turbo TTS wrapper for voice responses"""

    def __init__(
        self,
        voice_sample: Optional[Path] = None,
        use_paralinguistics: bool = True,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize TTS client

        Args:
            voice_sample: Path to voice sample for cloning (optional)
            use_paralinguistics: Enable [laugh], [chuckle], etc. tags
            output_dir: Directory for generated audio files
        """
        self.voice_sample = voice_sample
        self.use_paralinguistics = use_paralinguistics
        self.output_dir = output_dir or Path(tempfile.gettempdir())
        self._model = None
        self._lock = asyncio.Lock()

    async def _load_model(self):
        """Lazy load Chatterbox model"""
        global _chatterbox_model

        if _chatterbox_model is not None:
            self._model = _chatterbox_model
            return

        async with self._lock:
            if self._model is not None:
                return

            logger.info("Loading Chatterbox-Turbo model...")

            try:
                loop = asyncio.get_event_loop()

                def _load():
                    from chatterbox.tts import ChatterboxTTS
                    return ChatterboxTTS.from_pretrained(device="cuda")

                self._model = await loop.run_in_executor(None, _load)
                _chatterbox_model = self._model
                logger.info("Chatterbox model loaded (~350MB VRAM)")

            except ImportError as e:
                logger.error(f"Chatterbox not installed: {e}")
                logger.info("Install with: pip install chatterbox-tts")
                raise
            except Exception as e:
                logger.error(f"Failed to load Chatterbox: {e}")
                raise

    def _add_paralinguistics(self, text: str) -> str:
        """
        Add paralinguistic tags based on text content

        Chatterbox supports: [laugh], [chuckle], [sigh], [cough], [sniffle],
                            [groan], [yawn], [gasp]
        """
        if not self.use_paralinguistics:
            return text

        # Simple heuristics for adding tags
        # User can also manually include tags in waifu responses

        # Add [chuckle] for light teasing
        if any(word in text.lower() for word in ["hehe", "fufu", "ehehe"]):
            text = text.replace("hehe", "[chuckle]")
            text = text.replace("fufu", "[chuckle]")
            text = text.replace("ehehe", "[chuckle]")

        # Add [sigh] for concern
        if any(word in text.lower() for word in ["sigh", "haah"]):
            text = text.replace("sigh", "[sigh]")
            text = text.replace("haah", "[sigh]")

        return text

    def _clean_for_tts(self, text: str) -> str:
        """Clean text for TTS processing"""
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
        text = re.sub(r'`([^`]+)`', r'\1', text)       # `code`

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Clean up whitespace
        text = ' '.join(text.split())

        return text.strip()

    async def generate_speech(
        self,
        text: str,
        output_path: Optional[Path] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5
    ) -> Path:
        """
        Generate speech audio from text

        Args:
            text: Text to speak
            output_path: Custom output path (optional)
            exaggeration: Emotion exaggeration (0.0-1.0)
            cfg_weight: CFG guidance weight

        Returns:
            Path to generated audio file (WAV)
        """
        await self._load_model()

        # Clean and prepare text
        text = self._clean_for_tts(text)
        text = self._add_paralinguistics(text)

        if not text:
            raise ValueError("Empty text after cleaning")

        logger.debug(f"Generating speech for: {text[:50]}...")

        # Generate output path if not provided
        if output_path is None:
            import uuid
            output_path = self.output_dir / f"tts_{uuid.uuid4().hex[:8]}.wav"

        loop = asyncio.get_event_loop()

        def _generate():
            # Generate audio
            wav = self._model.generate(
                text,
                audio_prompt=str(self.voice_sample) if self.voice_sample else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )

            # Save to file
            import torchaudio
            torchaudio.save(str(output_path), wav, self._model.sr)

            return output_path

        result = await loop.run_in_executor(None, _generate)
        logger.info(f"Generated speech: {result}")
        return result

    async def convert_to_ogg(self, wav_path: Path) -> Path:
        """
        Convert WAV to OGG for Telegram voice messages

        Args:
            wav_path: Path to WAV file

        Returns:
            Path to OGG file
        """
        ogg_path = wav_path.with_suffix('.ogg')

        loop = asyncio.get_event_loop()

        def _convert():
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(str(wav_path))
            audio.export(str(ogg_path), format='ogg', codec='libopus')
            return ogg_path

        result = await loop.run_in_executor(None, _convert)
        logger.debug(f"Converted to OGG: {result}")
        return result

    async def text_to_voice_message(
        self,
        text: str,
        exaggeration: float = 0.5
    ) -> Path:
        """
        Generate speech and convert to Telegram voice message format

        Args:
            text: Text to speak
            exaggeration: Emotion exaggeration

        Returns:
            Path to OGG file ready for Telegram
        """
        # Generate WAV
        wav_path = await self.generate_speech(text, exaggeration=exaggeration)

        # Convert to OGG for Telegram
        ogg_path = await self.convert_to_ogg(wav_path)

        # Clean up WAV
        try:
            wav_path.unlink()
        except Exception:
            pass

        return ogg_path

    def unload_model(self):
        """Unload model from VRAM"""
        global _chatterbox_model

        if self._model is not None:
            logger.info("Unloading Chatterbox model")
            del self._model
            self._model = None
            _chatterbox_model = None

            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


# Global singleton
_tts_client: Optional[TTSClient] = None


def get_tts_client(
    voice_sample: Optional[Path] = None,
    use_paralinguistics: bool = True,
    output_dir: Optional[Path] = None
) -> TTSClient:
    """Get or create global TTS client"""
    global _tts_client

    if _tts_client is None:
        _tts_client = TTSClient(
            voice_sample=voice_sample,
            use_paralinguistics=use_paralinguistics,
            output_dir=output_dir
        )

    return _tts_client
