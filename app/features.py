import math


def temporal_features(value) -> list[float]:
    """Codificación temporal compartida por entrenamiento e inferencia."""
    timestamp = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    slot = int(timestamp.timestamp() // 900)
    return [
        math.sin(2 * math.pi * (slot % 96) / 96),
        math.cos(2 * math.pi * (slot % 96) / 96),
        math.sin(2 * math.pi * (slot % 672) / 672),
        math.cos(2 * math.pi * (slot % 672) / 672),
    ]
