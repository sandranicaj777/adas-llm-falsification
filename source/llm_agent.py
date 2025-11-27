import requests

USE_FAKE_LLM = True


class LLMAgent:
    def __init__(self, model="mistral", url="http://localhost:11434"):
        self.model = model
        self.url = f"{url}/api/generate"
        print(f"[LLMAgent] Using model '{self.model}' at {self.url}")

    def get_action(self, prompt: str) -> str:

        if USE_FAKE_LLM:
            return "BRAKE_LIGHT"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }

        try:
            resp = requests.post(self.url, json=payload, timeout=20)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip().upper()

            if "EMERGENCY_BRAKE" in text: return "EMERGENCY_BRAKE"
            if "BRAKE_LIGHT" in text: return "BRAKE_LIGHT"
            if "ACCELERATE" in text: return "ACCELERATE"
            if "MAINTAIN_SPEED" in text: return "MAINTAIN_SPEED"
            return "BRAKE_LIGHT"

        except Exception as e:
            print(f"[LLMAgent] Error calling LLM: {e}. Using EMERGENCY_BRAKE.")
            return "EMERGENCY_BRAKE"
