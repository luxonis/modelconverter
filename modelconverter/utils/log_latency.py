import re

import depthai as dai

RVC2_INFERENCE_LATENCY_RE = re.compile(
    r"NeuralNetwork inference took\s+'(?P<latency_ms>[0-9]+(?:\.[0-9]+)?)'\s+ms"
)
RVC4_INFERENCE_LATENCY_RE = re.compile(
    r"Executing the model took:\s*(?P<latency_ms>[0-9]+(?:\.[0-9]+)?)\s*ms"
)


def parse_inference_latency(
    log_message: dai.LogMessage, pattern: re.Pattern[str]
) -> float | None:
    """Extract an inference duration in milliseconds from a device log
    message."""
    payload = log_message.payload
    if isinstance(payload, bytes):
        payload = payload.decode(errors="replace")

    match = pattern.search(str(payload))
    return float(match.group("latency_ms")) if match else None
