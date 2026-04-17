import re

RELEVANCE_WORDS = [
    "AI", "Artificial Intelligence", "NVIDIA", "OpenAI", "Semiconductor",
    "TSMC", "IPO", "Layoff", "Cloud", "Computing", "Machine Learning",
    "Cybersecurity", "Startup", "Venture Capital", "Big Tech", "Regulation",
    "Quantum", "Automation", "SaaS", "Fintech"
]
RELEVANCE_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in RELEVANCE_WORDS) + r")\b", re.IGNORECASE)

headlines = [
    "Sora AI CTO Malfunctions After Being Asked Basic Question [video]",
    "Why some vehicles are set to lose access to carpool lanes in California",
    "Claude 3 overtakes GPT-4 in the duel of the AI bots",
    "Can GPT optimize my taxes? An experiment in letting the LLM be the UX"
]

for h in headlines:
    print(f"'{h}': {bool(RELEVANCE_RE.search(h))}")

# Check what matched in the failing one
h_fail = "Why some vehicles are set to lose access to carpool lanes in California"
match = RELEVANCE_RE.search(h_fail)
if match:
    print(f"Matched: '{match.group(0)}'")
else:
    print("No match found.")
