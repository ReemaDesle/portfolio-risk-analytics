import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class LLMNarrator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("Warning: GROQ_API_KEY not found in environment. LLM narration will be unavailable.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
        
        self.model = "llama-3.1-8b-instant"

    def narrate_advisory(self, 
                        portfolio_type: str, 
                        action: str, 
                        shock_prob: float, 
                        risk_level: str, 
                        top_drivers: List[str]) -> str:
        """
        Generate a concise, data-driven justification for the portfolio advisory.
        
        Args:
            portfolio_type: e.g., 'tech', 'geopolitical'
            action: e.g., 'REDUCE / HEDGE', 'BUY / ADD'
            shock_prob: Probability from M1
            risk_level: Label from M3
            top_drivers: List of human-friendly feature names contributing to the signal
        """
        if not self.client:
            return ""

        drivers_str = ", ".join(top_drivers) if top_drivers else "market baseline"
        
        prompt = f"""
        System: You are an expert financial risk advisor for retail investors. 
        Context:
        - Portfolio Type: {portfolio_type}
        - Recommended Action: {action}
        - Shock Probability: {shock_prob*100:.1f}%
        - Risk Level: {risk_level}
        - Key Drivers: {drivers_str}

        Task: 
        Provide a 2-sentence justification for this advice in plain, simple English.
        Explain how the 'Key Drivers' are causing this risk. 
        Avoid technical jargon or big words. Be direct and helpful.
        Tone: Empathetic but firm. Grade level target: 7th grade.

        Output format: Just the 2 sentences of text.
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return ""

# Singleton instance
narrator = LLMNarrator()
