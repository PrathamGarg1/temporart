"""Bedrock client that writes a reimbursement row on every call."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "aws_train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_ledger import log_call  # noqa: E402
from codemix_convert_bedrock import BedrockClient  # noqa: E402

THROTTLE_HINTS = (
    "429",
    "throttl",
    "ratelimit",
    "rate limit",
    "too many",
    "timeout",
    "timed out",
    "connection",
    "503",
    "502",
    "overloaded",
)


def _usage_from_openai(resp) -> tuple[int, int, str]:
    usage = getattr(resp, "usage", None)
    inn = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    rid = getattr(resp, "id", "") or ""
    return inn, out, str(rid)


def _usage_from_converse(resp: dict) -> tuple[int, int, str]:
    usage = resp.get("usage") or {}
    inn = int(usage.get("inputTokens") or 0)
    out = int(usage.get("outputTokens") or 0)
    rid = (resp.get("ResponseMetadata") or {}).get("RequestId", "")
    return inn, out, str(rid)


class LoggedBedrock(BedrockClient):
    def __init__(self, region: str, model: str, backend: str, job: str) -> None:
        super().__init__(region, model, backend=backend)
        self.job = job

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.35,
        extra_messages: list[dict] | None = None,
    ) -> str:
        use_mantle = self.backend == "mantle" or (
            self.backend == "auto" and self._mantle_ok and self._mantle is not None
        )
        if use_mantle:
            reinit_done = False
            for attempt in range(7):
                try:
                    self._refresh_mantle()
                    messages = [{"role": "system", "content": system}]
                    if extra_messages:
                        messages.extend(extra_messages)
                    messages.append({"role": "user", "content": user})
                    assert self._mantle is not None
                    resp = self._mantle.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    inn, out, rid = _usage_from_openai(resp)
                    content = resp.choices[0].message.content
                    text = "" if content is None else content.strip().strip('"').strip("'")
                    log_call(
                        job=self.job,
                        service="bedrock-mantle",
                        model=self.model,
                        input_tokens=inn,
                        output_tokens=out,
                        request_id=rid,
                        ok=True,
                    )
                    return text
                except Exception as e:
                    name = type(e).__name__
                    blob = f"{name} {e}".lower()
                    authish = "auth" in blob or "unauthorized" in blob or "expired" in blob
                    log_call(
                        job=self.job,
                        service="bedrock-mantle",
                        model=self.model,
                        input_tokens=0,
                        output_tokens=0,
                        ok=False,
                        notes=f"{name}:{e}"[:240],
                    )
                    if authish and not reinit_done:
                        reinit_done = True
                        self._token_ts = 0.0
                        try:
                            self._init_mantle()
                            self._mantle_ok = True
                            continue
                        except Exception:
                            pass
                    if any(h in blob for h in THROTTLE_HINTS):
                        time.sleep(min(40.0, 1.5 * (2**attempt)) + random.random())
                        continue
                    self._mantle_ok = False
                    self.backend = "converse"
                    break

        converse_model = self.model
        if converse_model.startswith(("qwen.", "google.", "mistral.")) or "/" in converse_model:
            converse_model = "qwen.qwen3-32b-v1:0"
        try:
            resp = self._bedrock.converse(
                modelId=converse_model,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            inn, out, rid = _usage_from_converse(resp)
            text = resp["output"]["message"]["content"][0]["text"].strip().strip('"').strip("'")
            log_call(
                job=self.job,
                service="bedrock-runtime",
                model=converse_model,
                input_tokens=inn,
                output_tokens=out,
                request_id=rid,
                ok=True,
            )
            return text
        except Exception as e:
            log_call(
                job=self.job,
                service="bedrock-runtime",
                model=converse_model,
                input_tokens=0,
                output_tokens=0,
                ok=False,
                notes=f"{type(e).__name__}:{e}"[:240],
            )
            raise
