# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import inspect
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
from .client_utils import log
from .models import loaders, MimiModel, LMModel, LMGen
from .models.hybrid_prompt import wrap_with_system_tags
from .run_inference import get_condition_tensors
from .inference_utils.rag_manager import RAGManager
from .inference_utils.turn_manager import TurnManager
from .inference_utils.utils import get_conditioning_remote_async
from .reference import LLMReferenceGenerator


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class ServerState:
    model_type: str
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, model_type: str, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, cfg_coef: float, device: str | torch.device,
                 voice_prompt_dir: str | None = None,
                 default_text_prompt: str = "",
                 default_voice_prompt: str = "",
                 **kwargs):
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size=1, cfg_coef=cfg_coef)
        force_streaming_sum = kwargs.pop("force_streaming_sum", lm.rag_token_id is not None)
        self.lm_gen = LMGen(
            lm,
            cfg_coef=cfg_coef,
            condition_tensors=condition_tensors,
            sample_rate=int(mimi.sample_rate),
            frame_rate=float(mimi.frame_rate),
            force_streaming_sum=force_streaming_sum,
            **kwargs,
        )

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()
        self.voice_prompt_dir = voice_prompt_dir
        self.default_text_prompt = default_text_prompt
        self.default_voice_prompt = default_voice_prompt
        self.reference_encoder_url = os.environ.get("REFERENCE_ENCODER_URL")
        self.rag_enabled = lm.rag_token_id is not None and bool(self.reference_encoder_url)
        self.stt_wait_steps = int(0.5 * mimi.frame_rate)

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self):
        for chunk in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])

        torch.cuda.synchronize()

    def _resolve_voice_path(self, filename: str | None) -> str | None:
        if not filename:
            return None
        if self.voice_prompt_dir:
            return os.path.join(self.voice_prompt_dir, filename)
        return filename

    def apply_hybrid_prompt(self, text_prompt: str, voice_filename: str | None) -> None:
        if text_prompt:
            self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(
                wrap_with_system_tags(text_prompt)
            )
        else:
            self.lm_gen.text_prompt_tokens = None
        voice_path = self._resolve_voice_path(voice_filename)
        if voice_path:
            self.lm_gen.load_voice_prompt_path(voice_path)
        else:
            self.lm_gen.voice_prompt = None
            self.lm_gen.voice_prompt_audio = None
            self.lm_gen.voice_prompt_codes = None

    async def decode_and_send(
        self,
        tokens: torch.Tensor,
        ws: web.WebSocketResponse,
        opus_writer: sphn.OpusStreamWriter,
        *,
        rag_manager: RAGManager | None = None,
        turn_manager: TurnManager | None = None,
        task_group: asyncio.TaskGroup | None = None,
    ):
        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
        main_pcm = self.mimi.decode(tokens[:, 1:])
        main_pcm = main_pcm.cpu()
        opus_bytes = opus_writer.append_pcm(main_pcm[0, 0].numpy())
        if len(opus_bytes) > 0:
            await ws.send_bytes(b"\x01" + opus_bytes)
        text_token = tokens[0, 0, 0].item()
        rag_id = self.lm_gen.lm_model.rag_token_id
        if rag_manager is not None and rag_id is not None and text_token == rag_id:
            log("info", "[RAG] model emitted RAG token")
            if turn_manager is not None:
                turn_manager.handle_spoken_text(model_text="[RET]")
            await ws.send_bytes(b"\x02" + b"[RET]")
            if task_group is not None:
                await rag_manager.trigger(
                    task_group=task_group,
                    wait_steps=self.stt_wait_steps,
                    handle_reference_fn=self._handle_reference_text,
                    context_provider=(turn_manager.get_context if turn_manager is not None else None),
                )
        elif text_token not in (0, 3):
            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
            _text = _text.replace("▁", " ")
            if turn_manager is not None:
                turn_manager.handle_spoken_text(model_text=_text)
            msg = b"\x02" + bytes(_text, encoding="utf8")
            log("info", f"text token '{_text}'")
            await ws.send_bytes(msg)
        if rag_manager is not None:
            rag_manager.step()

    async def _handle_reference_text(self, reference_text: str | None, lm_label: str = "") -> None:
        if not reference_text or not self.reference_encoder_url:
            return
        streaming_sum_tensor = await get_conditioning_remote_async(
            text=reference_text,
            encoder_url=self.reference_encoder_url,
        )
        per_slot: list[torch.Tensor | None] = [streaming_sum_tensor.squeeze(0)]
        self.lm_gen.update_streaming_sum_tensors(per_slot)
        log("info", f"[RAG] injected streaming_sum {tuple(streaming_sum_tensor.shape)} lm={lm_label!r}")

    async def recv_loop(
        self,
        ws: web.WebSocketResponse,
        opus_reader: sphn.OpusStreamReader,
        opus_writer: sphn.OpusStreamWriter,
        *,
        rag_manager: RAGManager | None = None,
        turn_manager: TurnManager | None = None,
        task_group: asyncio.TaskGroup | None = None,
    ):
        all_pcm_data = None
        skip_frames = 1
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    log("error", f"{ws.exception()}")
                    break
                elif message.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    log("error", f"unexpected message type {message.type}")
                    continue
                message = message.data
                if not isinstance(message, bytes):
                    log("error", f"unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    log("warning", "empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    pcm = opus_reader.append_bytes(payload)
                    if pcm.shape[-1] == 0:
                        continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        be = time.time()
                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        chunk = torch.from_numpy(chunk)
                        chunk = chunk.to(device=self.device)[None, None]
                        codes = self.mimi.encode(chunk)
                        if skip_frames:
                            # The first input audio frame is ignored, as from the point of
                            # view of the model it is in the past. We still `mimi.encode` for simplicity,
                            # however as the first encoded frame has a specific structure (due to the left padding),
                            # we reset the streaming state of the encoder to reapply the padding on the next call.
                            self.mimi.reset_streaming()
                            skip_frames -= 1
                        for c in range(codes.shape[-1]):
                            tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                            if tokens is None:
                                continue
                            await self.decode_and_send(
                                tokens, ws, opus_writer,
                                rag_manager=rag_manager,
                                turn_manager=turn_manager,
                                task_group=task_group,
                            )
                        log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")
                else:
                    log("warning", f"unknown message kind {kind}")
        finally:
            log("info", "connection closed")

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        log("info", "accepted connection")

        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            text_prompt = request.query.get("text_prompt", self.default_text_prompt) or ""
            voice_prompt = request.query.get("voice_prompt", self.default_voice_prompt) or ""
            self.lm_gen.reset_generation_state()
            self.mimi.reset_streaming()
            self.apply_hybrid_prompt(text_prompt, voice_prompt)
            if self.lm_gen.has_hybrid_prompt:
                log("info", f"hybrid prompt text={text_prompt!r} voice={voice_prompt!r}")
                self.lm_gen.step_system_prompts(self.mimi)
                self.mimi.reset_streaming()
            await ws.send_bytes(b"\x00")
            turn_manager = TurnManager() if self.rag_enabled else None
            if self.rag_enabled:
                async with RAGManager(LLMReferenceGenerator()) as rag_manager:
                    async with asyncio.TaskGroup() as tg:
                        await self.recv_loop(
                            ws, opus_reader, opus_writer,
                            rag_manager=rag_manager,
                            turn_manager=turn_manager,
                            task_group=tg,
                        )
            else:
                await self.recv_loop(ws, opus_reader, opus_writer)
        log("info", "done with connection")
        return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
    parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
    parser.add_argument("--cfg-coef", type=float, default=1., help="CFG coefficient.")
    parser.add_argument("--text-prompt", type=str, default="", help="Default Hybrid System Prompt role text.")
    parser.add_argument("--voice-prompt", type=str, default="", help="Default voice prompt filename.")
    parser.add_argument("--voice-prompt-dir", type=str, default=None, help="Directory of voice prompts.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--no_fuse_lora", action="store_false", dest="fuse_lora", default=True,
                        help="Do not fuse LoRA layers intot Linear layers.")
    parser.add_argument("--half", action="store_const", const=torch.float16, default=torch.bfloat16,
                        dest="dtype", help="Run inference with float16, not bfloat16, better for old GPUs.")
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log("error", "Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer,
        lora_weights=args.lora_weight, config_path=args.config_path)
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "loading moshi")
    skip_conditioners = ["reference_with_time"] if os.environ.get("REFERENCE_ENCODER_URL") else []
    lm = checkpoint_info.get_moshi(
        device=args.device, dtype=args.dtype, fuse_lora=args.fuse_lora,
        skip_conditioners=skip_conditioners,
    )
    log("info", "moshi loaded")

    state = ServerState(
        checkpoint_info.model_type, mimi, text_tokenizer, lm, args.cfg_coef, args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        default_text_prompt=args.text_prompt,
        default_voice_prompt=args.voice_prompt,
        **checkpoint_info.lm_gen_config,
    )
    log("info", "warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    static_path: None | str = None
    if args.static is None:
        log("info", "retrieving the static content")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
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

        log("info", f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None, **tunnel_kwargs)  # type: ignore
        log("info", f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
        log("info", "Note that this tunnel goes through the US and you might experience high latency in Europe.")
    web.run_app(app, host=args.host , port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
