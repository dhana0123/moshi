# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""FullDuplexBench duplex judges (pause, backchannel, turn-taking, interruption)."""

from __future__ import annotations

import importlib.resources
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torchaudio
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon
from silero_vad import get_speech_timestamps, load_silero_vad

from ..llm import LLMClient
from .base import Judge


def check_TO(
    timestamps: List[Dict[str, Any]], duration_threshold: float, num_words_threshold: int
) -> bool:
    """Take-over if speech span >= duration_threshold or word count >= threshold."""
    if len(timestamps) == 0:
        return False

    prev_end = -1
    for timestamp_dict in timestamps:
        assert timestamp_dict["timestamp"][0] >= prev_end
        assert timestamp_dict["timestamp"][1] >= timestamp_dict["timestamp"][0]
        prev_end = timestamp_dict["timestamp"][1]

    duration = timestamps[-1]["timestamp"][1] - timestamps[0]["timestamp"][0]
    if duration >= duration_threshold:
        return True
    if len(timestamps) >= num_words_threshold:
        return True
    return False


def _load_icc_gt_distribution() -> Dict[str, Any]:
    ref = importlib.resources.files("moshi.evaluate.judge.assets").joinpath(
        "icc_gt_distribution.json"
    )
    with ref.open("r", encoding="utf-8") as f:
        return json.load(f)


class BackChannelJudge(Judge):
    WINDOW_SIZE = 0.2
    EPSILON = 1e-10
    TURN_DURATION_THRESHOLD = 1
    TURN_NUM_WORDS_THRESHOLD = 3

    def __init__(self):
        self.gt_distribution = _load_icc_gt_distribution()
        self.vad_model = load_silero_vad()

    def __call__(
        self, input_path: Path, path: Path, timestamps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        out_wav_path = path.with_suffix(".wav")

        wav, sr = torchaudio.load(out_wav_path)
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        max_end_time = wav.shape[-1] / 16000

        segments = get_speech_timestamps(
            wav,
            self.vad_model,
            return_seconds=True,
        )

        backchannel_prediction = []
        TO = False
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]

            segment_timestamps = []
            for timestamp_dict in timestamps:
                t_start = timestamp_dict["timestamp"][0]
                t_end = timestamp_dict["timestamp"][1]

                if t_start is None or (t_end is None and t_start > end_time):
                    continue

                if t_end is None:
                    t_end = t_start

                if t_start >= start_time and t_end <= end_time:
                    pass
                elif t_start <= end_time and t_end > end_time:
                    t_end = end_time
                elif t_start <= start_time and t_end > start_time:
                    t_start = start_time
                else:
                    continue

                segment_timestamps.append(
                    {"text": timestamp_dict["text"], "timestamp": [t_start, t_end]}
                )

            this_TO = check_TO(
                segment_timestamps,
                self.TURN_DURATION_THRESHOLD,
                self.TURN_NUM_WORDS_THRESHOLD,
            )
            if not this_TO:
                backchannel_prediction.append((start_time, end_time))

            TO = TO or this_TO

        freq = len(backchannel_prediction) / max_end_time if max_end_time > 0 else 0.0

        spk = path.stem.split("_")[-1]
        gt_distribution = self.gt_distribution[spk]
        js_divergence = self.get_js_divergence(
            backchannel_prediction, max_end_time, gt_distribution
        )

        return {
            "TO": TO,
            "js_divergence": js_divergence,
            "freq": freq,
        }

    def get_js_divergence(
        self,
        backchannel_prediction: List[Tuple[float, float]],
        max_end_time: float,
        gt_distribution: List[float],
    ) -> float:
        if len(backchannel_prediction) == 0:
            return 1.0

        time_intervals = [0 for _ in range(int(max_end_time / self.WINDOW_SIZE) + 1)]

        for interval in backchannel_prediction:
            start = int(interval[0] / self.WINDOW_SIZE)
            end = int(interval[1] / self.WINDOW_SIZE)
            for i in range(start, end + 1):
                if i < len(time_intervals):
                    time_intervals[i] += 1

        time_intervals_arr = np.array(time_intervals, dtype=np.float64)
        time_intervals_arr = time_intervals_arr + self.EPSILON
        time_intervals_arr = time_intervals_arr / np.sum(time_intervals_arr)
        time_intervals_list = list(time_intervals_arr)

        x_gt = np.linspace(0, 1, len(gt_distribution))
        x_pred = np.linspace(0, 1, len(time_intervals_list))

        interp_func = interp1d(
            x_gt, gt_distribution, kind="linear", fill_value="extrapolate"
        )
        gt_dist_resized = interp_func(x_pred)

        return float(jensenshannon(np.array(time_intervals_list), np.array(gt_dist_resized)))


class PauseHandlingJudge(Judge):
    TURN_DURATION_THRESHOLD = 1
    TURN_NUM_WORDS_THRESHOLD = 3

    def __call__(
        self, input_path: Path, path: Path, timestamps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "TO": check_TO(
                timestamps, self.TURN_DURATION_THRESHOLD, self.TURN_NUM_WORDS_THRESHOLD
            ),
        }


class TurnTakingJudge(Judge):
    TURN_DURATION_THRESHOLD = 1
    TURN_NUM_WORDS_THRESHOLD = 3

    def __call__(
        self, input_path: Path, path: Path, timestamps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        with open(input_path, "r", encoding="utf-8") as f:
            input_turn = json.load(f)["turn_taking"]

        TO = check_TO(
            timestamps, self.TURN_DURATION_THRESHOLD, self.TURN_NUM_WORDS_THRESHOLD
        )
        latency = -1.0
        if TO:
            input_end_time = input_turn[0]["timestamp"][0]
            latency = max(0.0, timestamps[0]["timestamp"][0] - input_end_time)

        return {
            "TO": TO,
            "latency": latency,
        }


class UserInterruptionJudge(Judge):
    TURN_DURATION_THRESHOLD = 1
    TURN_NUM_WORDS_THRESHOLD = 3
    SYSTEM_PROMPT = """
   The scenario is that the user and AI are talking in the spoken conversation.
   The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
   Your task is to rate the quality of AI's response after the user interrupt the turn.


   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
   - 0: The AI's response is totally unrelated to the user's interrupting turn.
   - 1: The AI's response is not related to the user's interrupting turn.
   - 2: The AI's response is slightly related to the user's interrupting turn.
   - 3: The AI's response is related to the user's interrupting turn.
   - 4: The AI's response is highly related to the user's interrupting turn.
   - 5: The AI's response is perfectly related to the user's interrupting turn.


   Firstly, briefly analyze the user's interrupting turn and the AI's response
   Then, you must return the overall output as the following format:
   Analysis: [Your analysis].
   I would rate the AI's response as [Rating].
   """

    def __init__(self, llm: LLMClient | None = None):
        import os

        if llm is None:
            self.llm = LLMClient(system_prompt=self.SYSTEM_PROMPT)
            self.llm.model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4-turbo")
        else:
            self.llm = llm

    def __call__(
        self, input_path: Path, path: Path, timestamps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        with open(input_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)["interrupt"]

            in_interrupt_text = metadata[0]["interrupt"]
            in_before_interrupt_text = metadata[0]["context"]
            input_end_time = metadata[0]["timestamp"][1]

        timestamps_after_interrupt = [
            timestamp_dict
            for timestamp_dict in timestamps
            if timestamp_dict["timestamp"][0] >= input_end_time
        ]
        out_after_interrupt_text = " ".join(
            [timestamp_dict["text"] for timestamp_dict in timestamps_after_interrupt]
        )

        TO = check_TO(
            timestamps_after_interrupt,
            self.TURN_DURATION_THRESHOLD,
            self.TURN_NUM_WORDS_THRESHOLD,
        )
        latency = -1.0
        score = -1
        if TO:
            latency = max(
                0.0, timestamps_after_interrupt[0]["timestamp"][0] - input_end_time
            )

            prompt = f"""
            - Contextual user turn: {in_before_interrupt_text}
            - User interrupting turn: {in_interrupt_text}
            - AI's response: {out_after_interrupt_text}
            """

            response = self.llm.generate(
                prompt=prompt, context="", max_new_tokens=512, stop_token=None
            )
            score = self._parse_response(response)

        return {
            "TO": TO,
            "latency": latency,
            "score": score,
        }

    def _parse_response(self, response: str | None) -> int:
        if response is None:
            return -1
        example_pattern = re.compile(
            r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)", re.DOTALL
        )

        rating = -1
        for match in example_pattern.finditer(response):
            rating = int(match.group(2).strip())

        return rating
