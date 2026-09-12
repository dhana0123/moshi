import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from moshi.train.data.inner_monologue import (
    JOINT_STREAMS,
    apply_text_delay,
    stack_joint_sequence,
)
from moshi.train.data.interleaver import Interleaver
from moshi.train.data.prepare import (
    alignments_payload,
    make_dummy_dialogue,
    pack_stereo,
    save_clip,
)
from moshi.train.data import prepare as prepare_mod
from moshi.train.data import asr as asr_mod
from moshi.train.data import push as push_mod


class _FakeSPM:
    def encode(self, text, enable_sampling=False, alpha=None, nbest_size=None):
        if isinstance(text, list):
            return [[11, 12] for _ in text]
        word = text if isinstance(text, str) else str(text)
        return [11] if word else []

    def bos_id(self):
        return 1

    def eos_id(self):
        return 2


def test_text_delay_shifts_words_earlier():
    al = [("hi", (0.16, 0.32), "SPEAKER_MAIN")]
    delayed = apply_text_delay(al, 0.16)
    assert delayed[0][1][0] == 0.0
    assert delayed[0][1][1] == 0.16


def test_stack_is_17_streams_text_then_agent_then_user():
    t = 4
    text = torch.full((1, 1, t), 3)
    agent = torch.arange(8 * t).view(1, 8, t)
    user = torch.arange(8 * t, 16 * t).view(1, 8, t)
    joint = stack_joint_sequence(text, agent, user)
    assert joint.shape == (1, JOINT_STREAMS, t)
    assert torch.equal(joint[:, 0:1], text)
    assert torch.equal(joint[:, 1:9], agent)
    assert torch.equal(joint[:, 9:], user)


def test_pack_stereo_syncs_length():
    sr = 24000
    a = np.ones(sr, dtype=np.float32) * 0.1
    u = np.ones(sr // 2, dtype=np.float32) * 0.1
    stereo = pack_stereo(a, u, sr)
    assert stereo.shape == (2, sr)


def test_dummy_clip_writes_jsonl_and_agent_alignments(tmp_path: Path):
    stereo, words = make_dummy_dialogue()
    jsonl = tmp_path / "train.jsonl"
    clip = save_clip(tmp_path, "dummy", stereo, words, jsonl, text_delay_sec=0.16)
    assert clip.wav_path.exists()
    payload = json.loads(clip.json_path.read_text(encoding="utf-8"))
    assert payload["alignments"][0][2] == "SPEAKER_MAIN"
    # Acoustic times are stored unshifted; delay is train-time.
    assert payload["alignments"][0][1][0] == 0.2
    assert payload["inner_monologue"]["control_tokens"] is False
    row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert row["duration"] > 2.0


def test_interleaver_epad_on_dummy_words():
    tok = _FakeSPM()
    inter = Interleaver(
        tok,
        audio_frame_rate=12.5,
        text_padding=3,
        end_of_text_padding=0,
        zero_padding=-1,
        keep_main_only=True,
        device="cpu",
    )
    _, words = make_dummy_dialogue()
    alignments = [(w, ts, "SPEAKER_MAIN") for w, ts in words]
    stream = inter.prepare_item(alignments, segment_duration=2.56)
    ids = stream[0, 0].tolist()
    assert 0 in ids
    assert 11 in ids
    assert 3 in ids


def test_prepare_dummy_cli(tmp_path: Path):
    code = prepare_mod.main(["--out", str(tmp_path / "im"), "--dummy"])
    assert code == 0
    jsonl = tmp_path / "im" / "train.jsonl"
    meta = json.loads((tmp_path / "im" / "dataset_meta.json").read_text(encoding="utf-8"))
    assert jsonl.exists()
    assert meta["streams"] == 17
    assert meta["control_tokens"] is False
    assert meta["layout"][0] == "W_agent_text"
    assert meta["asr_backend"] == "indic-conformer"


def test_alignments_payload_no_user_text():
    payload = alignments_payload([("ok", (0.0, 0.2))])
    assert payload["inner_monologue"]["user_text"] is False


def test_even_word_spans_and_split():
    assert asr_mod.split_words("  namaste  bhai ") == ["namaste", "bhai"]
    spans = asr_mod._even_word_spans(["a", "b"], 1.0)
    assert spans[0] == ("a", (0.0, 0.5))
    assert spans[1] == ("b", (0.5, 1.0))


def test_transcribe_agent_words_conformer_mocked():
    mono = np.zeros(16000, dtype=np.float32)
    with (
        patch.object(asr_mod, "transcribe_indic_conformer", return_value="hello there") as tr,
        patch.object(
            asr_mod,
            "align_words",
            return_value=[("hello", (0.0, 0.4)), ("there", (0.4, 0.8))],
        ) as al,
    ):
        words = asr_mod.transcribe_agent_words(
            mono, 16000, language="hi", backend="indic-conformer", device="cpu"
        )
    tr.assert_called_once()
    al.assert_called_once()
    assert words[0][0] == "hello"
    assert words[1][1] == (0.4, 0.8)


def test_push_private_dataset_mocked(tmp_path: Path):
    out = tmp_path / "pack"
    out.mkdir()
    (out / "dataset_meta.json").write_text("{}", encoding="utf-8")
    wav_dir = out / "wav"
    wav_dir.mkdir()
    (wav_dir / "a.json").write_text('{"alignments":[]}', encoding="utf-8")
    abs_wav = (wav_dir / "a.wav").resolve()
    abs_wav.write_bytes(b"RIFF")
    (out / "train.jsonl").write_text(
        json.dumps({"path": str(abs_wav), "duration": 1.0}) + "\n",
        encoding="utf-8",
    )

    api = MagicMock()
    api.whoami.return_value = {"name": "testuser"}
    with patch("huggingface_hub.HfApi", return_value=api):
        repo = push_mod.push_private_dataset(out, hf_dataset=None, include_tokens=False)

    assert repo == "testuser/moshi-indic-inner-monologue"
    api.create_repo.assert_called_once()
    kwargs = api.create_repo.call_args.kwargs
    assert kwargs["private"] is True
    assert kwargs["repo_type"] == "dataset"
    api.upload_folder.assert_called_once()
    row = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["path"] == "wav/a.wav"
    assert (out / "README.md").exists()


def test_resolve_dataset_repo_override():
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    with patch("huggingface_hub.HfApi", return_value=api):
        assert push_mod.resolve_dataset_repo_id("bob/custom") == "bob/custom"
        assert push_mod.resolve_dataset_repo_id("shortname") == "alice/shortname"
