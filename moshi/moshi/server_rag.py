# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import inspect
import os
import signal
import tarfile
import secrets
import sys
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import aiohttp
import sentencepiece
import torch
from aiohttp import web
from huggingface_hub import hf_hub_download

from .models import loaders, MimiModel, LMGen
from .reference.llm_reference_generator import LLMReferenceGenerator
from .inference_utils import load_models
from .inference_utils.batch_runner import BatchInput, BatchRunner
from .inference_utils.utils import get_reference_encoder_url, seed_all, setup_logging
from .inference_utils.channel import Channel, StepOutput
from .inference_utils.retrieval_profiles import default_profile_id, load_retrieval_env

logger = logging.getLogger(__name__)


class _ColorFormatter(logging.Formatter):
    """Formatter with ANSI colors per level and centisecond timestamps."""

    _COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def __init__(self):
        super().__init__(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    def formatTime(self, record, datefmt=None):
        import datetime

        ct = datetime.datetime.fromtimestamp(record.created)
        s = ct.strftime("%H:%M:%S")
        return f"{s}.{int(record.msecs / 10):02d}"

    def format(self, record):
        color = self._COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname:<7}{self._RESET}"
        return super().format(record)


class ServerState:
    """Shared resources for every HTTP channel; owns a ``BatchRunner`` for GPU steps."""

    def __init__(
        self,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm_gen: LMGen,
        reference_encoder_url: str,
        device: str | torch.device | None = None,
        rag_timeout: float = 1.5,
        max_reference_tokens: int = 512,
        stt_wait_time: float = 0.5,
        gradium_stt: bool = False,
        batch_size: int = 16,
        power_threshold: float | None = None,
        vad_window_size: int = 4,
        vad_threshold: float = 0.5,
        init_active_speaker: str = "model",
    ):
        self.text_tokenizer = text_tokenizer
        self.reference_encoder_url = reference_encoder_url
        self.rag_timeout = rag_timeout
        self.max_reference_tokens = max_reference_tokens
        self.vad_window_size = vad_window_size
        self.vad_threshold = vad_threshold
        self.init_active_speaker = init_active_speaker
        self.power_threshold = power_threshold
        self.stt_wait_steps = int(stt_wait_time * mimi.frame_rate) if stt_wait_time > 0 else 0
        self.gradium_stt = gradium_stt
        self.mimi_copy = deepcopy(mimi)

        mimi.set_num_codebooks(lm_gen.lm_model.num_codebooks - 1)

        retrieval_env = load_retrieval_env()
        self._retrieval_profiles = retrieval_env.profiles
        n_prof = len(self._retrieval_profiles)
        if n_prof == 0:
            logger.info("[Retrieval] MOSHI_RETRIEVAL_LLMS_JSON unset or empty; using single LLM from env")
        elif n_prof == 1:
            logger.info(
                "[Retrieval] one entry in MOSHI_RETRIEVAL_LLMS_JSON (%r); need >=2 for UI switching, using env LLM",
                self._retrieval_profiles[0].id,
            )
        else:
            did = default_profile_id(self._retrieval_profiles)
            logger.info(
                "[Retrieval] loaded %d profiles ids=%s default(fallback+initial)=%r (WebSocket switching enabled)",
                n_prof,
                [p.id for p in self._retrieval_profiles],
                did,
            )

        # Shared LLM used for reference generation. Per-channel reference history
        # is owned by each channel's ``RAGManager``, so the model itself is stateless.
        if len(self._retrieval_profiles) >= 2:
            self.reference_generator = LLMReferenceGenerator(retrieval_profiles=self._retrieval_profiles)
        else:
            style: str = "original"
            if n_prof == 1:
                style = self._retrieval_profiles[0].prompt_style
            self.reference_generator = LLMReferenceGenerator(prompt_style=style)

        self.batch_size = batch_size
        self.device = device
        self.runner = BatchRunner(mimi, lm_gen, device, batch_size)
        self.frame_size = self.runner.frame_size

        # Per-slot occupant registry (``Channel`` or offline ``InferenceJob``), plus a lock.
        self.slots: list[Any | None] = [None] * self.batch_size
        self._slots_lock = asyncio.Lock()

    def warmup(self) -> None:
        self.runner.warmup()
        self.reference_generator.warmup()

    async def acquire_slot(self, occupant: Any) -> int:
        """Reserve a free batch slot. Raises ``web.HTTPServiceUnavailable`` if none is free."""
        async with self._slots_lock:
            for i, slot in enumerate(self.slots):
                if slot is None:
                    self.slots[i] = occupant
                    logger.info(
                        f"[Slot] acquired slot {i} ({sum(s is not None for s in self.slots)}/{self.batch_size})"
                    )
                    return i
            raise web.HTTPServiceUnavailable(reason="No available slots")

    async def wait_acquire_slot(self, occupant: Any) -> int:
        """Block until a slot is free, then reserve it and set ``occupant.slot_idx``."""
        while True:
            try:
                idx = await self.acquire_slot(occupant)
                occupant.slot_idx = idx
                return idx
            except web.HTTPServiceUnavailable:
                await asyncio.sleep(0.005)

    async def release_slot(self, slot_idx: int):
        """Release a slot. The actual streaming-state reset happens inside the
        step loop the next time the slot is re-used (``is_first`` frame),
        keeping all CUDA work in one place."""
        async with self._slots_lock:
            self.slots[slot_idx] = None
            logger.info(f"[Slot] released slot {slot_idx}")

    def _gather_step_inputs(self) -> BatchInput[Any] | None:
        filtered_pcm_batch = torch.zeros(
            self.batch_size,
            1,
            self.frame_size,
            dtype=torch.float32,
            device=self.device,
        )
        lm_mask_cpu = torch.zeros(self.batch_size, dtype=torch.bool)
        first_mask_cpu = torch.zeros(self.batch_size, dtype=torch.bool)
        active: list[tuple[int, Any]] = []
        for b, occupant in enumerate(self.slots):
            if occupant is None:
                continue
            try:
                inp = occupant.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
            filtered_pcm_batch[b, 0] = inp.filtered_pcm[0]
            lm_mask_cpu[b] = True
            if inp.is_first:
                first_mask_cpu[b] = True
            active.append((b, occupant))

        if not active:
            return None
        return BatchInput(
            filtered_pcm_batch=filtered_pcm_batch,
            lm_mask_cpu=lm_mask_cpu,
            first_mask_cpu=first_mask_cpu,
            active=active,
            pcm_batch=None,
        )

    def _deliver_step_row(
        self,
        occupant: Any,
        *,
        text_token: int,
        pcm_out: torch.Tensor | None,
    ) -> None:
        occupant.output_queue.put_nowait(StepOutput(text_token=text_token, pcm=pcm_out))

    async def run_one_step(self) -> bool:
        g = self._gather_step_inputs()
        if g is None:
            return False
        return self.runner.run_step(g, self._deliver_step_row)

    async def _step_loop(self):
        try:
            while True:
                ran = await self.run_one_step()
                if ran:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            logger.info("[Step] step loop cancelled")
            raise

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        logger.info("new WebSocket client connected")
        try:
            async with Channel(self, ws, mimi=deepcopy(self.mimi_copy)) as channel:
                await channel.run()
        except web.HTTPServiceUnavailable as e:
            logger.warning("Rejecting connection: %s", e.reason)
            raise
        logger.info("WebSocket client disconnected")
        return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action="store_true", help="Activate a gradio tunnel.")
    parser.add_argument(
        "--gradio-tunnel-token", help="Provide a custom (secret) token here to keep getting the same URL."
    )

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_RAG_REPO,
        help="HF repo to look into, defaults Moshiko. Use this to select a different pre-trained model.",
    )
    parser.add_argument("--config", type=str, help="Path to a local config file.", default=None)
    parser.add_argument("--cfg-coef", type=float, default=1.0, help="CFG coefficient.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for Moshi and Mimi (reference text uses the HTTP reference API, not a local model).",
    )
    parser.add_argument(
        "--stt-wait-time",
        type=float,
        default=1.0,
        help="Wait time for STT model to generate full transcript in seconds. "
        "Set it to a higher value when using DSM rather than Gradium with automatic flushing.",
    )
    parser.add_argument(
        "--gradium-stt",
        action="store_true",
        dest="gradium_stt",
        help="Use the official Gradium client with flushing. Will reduce stt_wait_time automatically.",
    )
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Run inference with float16, not bfloat16, better for old GPUs.",
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )
    parser.add_argument(
        "--rag-timeout",
        type=float,
        default=1.5,
        help="Timeout for reference text generation in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--max-reference-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for reference text (default: 512).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Maximum number of concurrent client conversations served by the batched step loop.",
    )
    parser.add_argument(
        "--vad-window-size",
        type=int,
        default=4,
        help="Number of consecutive VAD frames required for speaker transition (default: 4).",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD probability threshold for determining speaker (default: 0.5).",
    )
    parser.add_argument(
        "--power-threshold",
        type=int,
        default=-65,
        help="If set (in dB), zero-out input audio chunks below this RMS power before Mimi encode.",
    )
    parser.add_argument(
        "--init-active-speaker",
        type=str,
        default="model",
        choices=["model", "user"],
        help="Initial speaker used to fetch the initial condition tensors (default: model).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO).",
    )

    args = parser.parse_args()

    # Configure logging with colored output and centisecond timestamps.
    setup_logging(getattr(logging, args.log_level.upper()), _ColorFormatter())

    # Suppress aiohttp access logs (HTTP request logs)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

    # Get and validate REFERENCE_ENCODER_URL environment variable
    reference_encoder_url = get_reference_encoder_url()
    logger.info(f"Using Reference Encoder service at: {reference_encoder_url}")

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ""
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error(
                "Cannot find gradio which is required to activate a tunnel. Please install with `pip install gradio`."
            )
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Load all models
    mimi, text_tokenizer, lm_gen = load_models(args)

    state = ServerState(
        mimi=mimi,
        text_tokenizer=text_tokenizer,
        lm_gen=lm_gen,
        reference_encoder_url=reference_encoder_url,
        stt_wait_time=args.stt_wait_time,
        gradium_stt=args.gradium_stt,
        device=args.device,
        rag_timeout=args.rag_timeout,
        max_reference_tokens=args.max_reference_tokens,
        batch_size=args.batch_size,
        vad_window_size=args.vad_window_size,
        vad_threshold=args.vad_threshold,
        init_active_speaker=args.init_active_speaker,
        power_threshold=args.power_threshold,
    )
    logger.info("warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    async def handle_health(_: web.Request) -> web.Response:
        """GET /api/health: health check endpoint, always returns OK."""
        return web.json_response({"status": "ok"})

    app.router.add_get("/api/health", handle_health)

    async def handle_session_feedback(request: web.Request) -> web.Response:
        """
        POST /api/session_feedback

        Forwards JSON to MOSHI_FEEDBACK_WEBHOOK_URL (e.g. a Google Apps Script web app).
        """
        webhook_url = os.getenv("MOSHI_FEEDBACK_WEBHOOK_URL", "").strip()
        if not webhook_url:
            logger.warning(
                "session_feedback: MOSHI_FEEDBACK_WEBHOOK_URL not set; accepting but not forwarding",
            )
            return web.Response(status=202, text="feedback webhook not configured")

        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, text="invalid json")

        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(webhook_url, json=body) as resp:
                    if 200 <= resp.status < 300:
                        return web.Response(status=204)
                    text = await resp.text()
                    logger.error(
                        f"session_feedback: webhook returned {resp.status}: {text}",
                    )
                    return web.Response(status=502, text="upstream webhook error")
        except Exception as e:
            logger.error(f"session_feedback: request failed: {e}")
            return web.Response(status=502, text="webhook request failed")

    app.router.add_post("/api/session_feedback", handle_session_feedback)

    static_path: None | str = None
    if args.static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("kyutai/moshi-rag-artifacts", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        static_path = str(dist)
    elif args.static != "none":
        # When set to the "none" string, we don't serve any static content.
        static_path = args.static
    if static_path is not None:

        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    async def on_shutdown(app):
        logger.info("[Shutdown] aiohttp app shutdown triggered")

    async def on_cleanup(app):
        logger.info("[Shutdown] aiohttp app cleanup triggered")

    app.on_shutdown.append(on_shutdown)
    app.on_cleanup.append(on_cleanup)

    logger.info(f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel("localhost", args.port, tunnel_token, None, **tunnel_kwargs)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
        logger.info("Note that this tunnel goes through the US and you might experience high latency in Europe.")

    async def _serve():
        """Run the HTTP server and the batched step loop under a single task group.

        Any background task failure (e.g. the step loop crashing) propagates out
        of the TaskGroup and terminates the whole server with its traceback,
        instead of being silently swallowed by a fire-and-forget task.
        """
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port, ssl_context=ssl_context)
        await site.start()
        logger.info("[Startup] HTTP server listening")

        # ``asyncio.run`` already translates SIGINT into cancellation of the
        # main task. Add SIGTERM so Docker/k8s termination uses the same path.
        main_task = asyncio.current_task()
        assert main_task is not None
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, main_task.cancel)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(state._step_loop(), name="batched-step-loop")
                # Block until cancelled (SIGINT/SIGTERM) or until a child task
                # raises (TaskGroup cancels this wait in that case).
                await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_serve())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("[Shutdown] interrupted")
    logger.info("server stopped")


if __name__ == "__main__":
    with torch.no_grad():
        main()
