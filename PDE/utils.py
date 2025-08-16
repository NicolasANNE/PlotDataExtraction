import base64

def encode_image(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def gemini_cost(usage_metadata) -> float:
    """Compute cost of Gemini API usage.
    Add cost for each model."""
    total = 0
    for model, usage in usage_metadata.items():
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        if model == "gemini-2.0-flash":
            cost_in = 0.10 * (input_tokens / 1_000_000)
            cost_out = 0.40 * (output_tokens / 1_000_000)
        else:
            raise ValueError(f"Unsupported model: {model}")
        total += cost_in + cost_out
    return total