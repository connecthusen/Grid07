
from langchain_core.tools import tool
from grid07.config import get_logger

log = get_logger(__name__)



_HEADLINES: dict[str, list[str]] = {

    # AI / LLM
    "ai": [
        "OpenAI releases GPT-5 with advanced reasoning capabilities",
        "Google DeepMind's Gemini Ultra beats humans on 90% of benchmarks",
        "AI tools replacing junior developers at major tech firms, study finds",
        "Sam Altman warns AGI could arrive within 3 years",
        "Anthropic raises $4B as AI arms race intensifies",
    ],
    "openai": [
        "OpenAI launches new model that codes, reasons, and browses the web",
        "OpenAI revenue hits $3.4B as enterprise adoption surges",
        "OpenAI in talks with US government over national AI infrastructure",
    ],
    "llm": [
        "Meta releases Llama 4 as fully open-source model",
        "Mistral overtakes GPT-4 on coding benchmarks",
        "LLM hallucination rates drop 60% with new training techniques",
    ],

    # Crypto / Blockchain
    "crypto": [
        "Bitcoin surges past $95,000 as institutional demand explodes",
        "SEC approves spot Ethereum ETF — markets rally hard",
        "Crypto market cap crosses $3 trillion for first time",
        "BlackRock Bitcoin ETF becomes largest in history with $25B AUM",
    ],
    "bitcoin": [
        "Bitcoin hits new all-time high amid regulatory ETF approvals",
        "MicroStrategy adds 10,000 BTC to treasury — stock up 12%",
        "Bitcoin dominance hits 58% as altcoins bleed",
    ],
    "ethereum": [
        "Ethereum staking yields rise to 6.2% post-merge upgrades",
        "ETH L2 networks process more transactions than Ethereum mainnet",
    ],

    # Finance / Markets
    "stock": [
        "S&P 500 hits record high on strong earnings season",
        "Tech stocks lead rally as Nasdaq gains 2.3% in a single session",
        "Hedge funds rotate out of bonds into equities amid rate cut hopes",
    ],
    "market": [
        "Wall Street bulls return as Fed signals end of rate hike cycle",
        "VIX drops to 12 — lowest volatility reading in 18 months",
        "Global markets up as China stimulus package exceeds expectations",
    ],
    "fed": [
        "Federal Reserve cuts rates by 25bps — first cut in 4 years",
        "Fed chair signals two more cuts possible before year end",
        "Treasury yields invert again as recession fears mount",
    ],
    "interest": [
        "Mortgage rates fall to 6.1% following Fed pivot",
        "High interest rates squeeze small business lending to decade low",
    ],
    "etf": [
        "Bitcoin ETF inflows hit $500M in single day — record breaking",
        "Spot Ethereum ETF approved — crypto markets surge 15%",
        "Gold ETF sees largest outflow in 5 years as crypto competes",
    ],

    # Big Tech / Monopoly
    "google": [
        "DOJ antitrust trial finds Google illegally monopolized search",
        "Google fined $5B by EU for anti-competitive Android practices",
        "Google lays off 12,000 workers while posting record ad revenue",
    ],
    "amazon": [
        "Amazon acquires iRobot raising antitrust alarm bells in Brussels",
        "Amazon warehouse workers unionize in three new states",
        "AWS dominates cloud with 32% market share — rivals struggle",
    ],
    "meta": [
        "Meta's ad revenue surges 27% — Zuckerberg touts AI targeting",
        "EU fines Meta $1.3B for illegal data transfers to US servers",
        "Meta secretly collects data on non-users, lawsuit alleges",
    ],
    "microsoft": [
        "Microsoft Copilot integrated into every Office product globally",
        "Microsoft-Activision deal faces new EU scrutiny over game market",
    ],

    # Privacy / Surveillance
    "privacy": [
        "New EU AI Act mandates transparency for all AI-generated content",
        "NSA bulk data collection program ruled unconstitutional",
        "Apple's App Tracking Transparency cuts Meta revenue by $10B",
    ],
    "surveillance": [
        "China's social credit system expands facial recognition to transit",
        "UK government proposes mass surveillance of private messages",
        "Clearview AI fined $10M for scraping billions of photos without consent",
    ],

    # Elon Musk / Tesla / SpaceX
    "elon": [
        "Elon Musk's xAI raises $6B to compete with OpenAI",
        "Musk claims Grok 3 will surpass all existing AI models by Q3",
        "Elon Musk fires 80% of Twitter staff — platform still running",
    ],
    "tesla": [
        "Tesla Full Self-Driving v13 achieves zero interventions in 1000-mile test",
        "Tesla Cybertruck production hits 1,000 units per week",
        "Tesla stock drops 8% after missing delivery targets",
    ],
    "spacex": [
        "SpaceX Starship completes first fully successful orbital test flight",
        "SpaceX wins $3.4B NASA contract for lunar lander mission",
        "Starlink surpasses 3 million subscribers globally",
    ],
}




@tool
def mock_searxng_search(query: str) -> str:
    query_lower = query.lower()
    matched: list[str] = []

    for keyword, headlines in _HEADLINES.items():
        if keyword in query_lower:
            for h in headlines:
                if h not in matched:       
                    matched.append(h)

    results = matched[:5] if matched else [
        "No recent headlines found. Use your existing knowledge to write the post."
    ]

    log.info("Search query: %r → %d headline(s) found", query, len(results))
    log.debug("Headlines: %s", results)

    return "\n".join(f"- {h}" for h in results)