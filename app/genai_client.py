import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("AIzaSyAMgX92z3SZ2OV-MyhotPpMWosF9OFr81c"))

def run_genai_reasoning(features: dict):
    """
    Uses Gemini as a reasoning layer (text-only, no vision).
    """

    prompt = f"""
    You are an image analysis reasoning system.

    Image metadata:
    - Edge density: {features['edge_density']}
    - Brightness: {features['brightness']}

    Task:
    Infer the most likely scene type.
    Explicitly mention uncertainty and confidence level.
    Keep the response concise and structured.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,   # lower = more deterministic
                "max_output_tokens": 200
            }
        )

        return {
            "summary": response.text.strip()
        }

    except Exception:
        # Hard fallback — no hallucinated confidence
        return {
            "summary": "Generic scene inference",
            "attributes": ["low confidence", "insufficient visual signal"]
        }
