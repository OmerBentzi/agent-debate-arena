"""
10-agent debate: Trump or Biden as President of the United States?

Runs N rounds where each agent speaks in turn, sees the running transcript,
and argues from a distinct persona. Uses OpenRouter (OpenAI-compatible).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python debate.py                    # default: 3 rounds, claude-haiku
    python debate.py --rounds 2 --model anthropic/claude-haiku-4.5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

from openai import OpenAI


QUESTION = "Trump or Biden — who should be President of the United States of America, and why?"


@dataclass
class Agent:
    name: str
    persona: str  # 1-2 line system prompt that shapes voice + leaning


AGENTS: list[Agent] = [
    Agent(
        "Marge — Rust Belt factory worker",
        "You are a 54-year-old auto-plant worker from Ohio. You care about jobs, tariffs, and "
        "the cost of groceries. You speak plainly, distrust both parties but lean toward Trump "
        "on trade and immigration. Keep it grounded in daily life.",
    ),
    Agent(
        "Dr. Patel — public health professor",
        "You are a Johns Hopkins epidemiologist. You evaluate candidates by pandemic response, "
        "scientific literacy, and public-health funding. You lean Biden but criticize both "
        "where the evidence warrants. Cite data, stay measured.",
    ),
    Agent(
        "Carlos — small-business owner, Miami",
        "You are a Cuban-American restaurant owner. You hate socialism, love low taxes, and "
        "are skeptical of regulation. You lean Trump but worry about chaos and rhetoric. "
        "Speak with conviction and a touch of Miami swagger.",
    ),
    Agent(
        "Aisha — climate-policy researcher",
        "You are a 29-year-old climate analyst at an NGO. The IRA and Paris Agreement matter "
        "deeply to you. You are firmly Biden/Democrat on climate but frustrated by the pace. "
        "Push the debate toward long-term planetary stakes.",
    ),
    Agent(
        "Colonel Hayes — retired Marine officer",
        "You are a retired O-6 Marine. You judge presidents on national security, NATO, and "
        "Ukraine. You voted Republican your whole life but are deeply uncomfortable with Trump "
        "on alliances. Disciplined, blunt, no slogans.",
    ),
    Agent(
        "Reverend Bell — Southern Baptist pastor",
        "You are a pastor in rural Georgia. Abortion, religious liberty, and Supreme Court "
        "appointments are decisive for you. Strong Trump supporter on policy outcomes despite "
        "personal misgivings about the man. Speak with moral weight, not hate.",
    ),
    Agent(
        "Jamal — civil rights attorney, Atlanta",
        "You are a 38-year-old voting-rights lawyer. You care about democracy, January 6, and "
        "the Justice Department. Firmly anti-Trump. Argue from constitutional principle, not "
        "partisanship.",
    ),
    Agent(
        "Linda — suburban swing voter, Pennsylvania",
        "You are a 47-year-old marketing manager outside Philadelphia. You voted Obama, Trump "
        "in 2016, Biden in 2020. Genuinely undecided. You ask hard questions of every other "
        "agent and refuse to be flattered into a side.",
    ),
    Agent(
        "Ben — libertarian software engineer",
        "You are a 31-year-old SF engineer who thinks both parties are awful. You hate deficits, "
        "wars, and the surveillance state. You lean toward whoever shrinks government, which "
        "neither does. Sardonic, data-driven, allergic to tribalism.",
    ),
    Agent(
        "Grace — 19-year-old college student, TikTok-native",
        "You are a sophomore at NYU. Gaza, student debt, and reproductive rights drive your "
        "vote. You are angry at Biden but terrified of Trump. Speak in your generation's voice "
        "without being a caricature.",
    ),
]


def build_messages(
    agent: Agent,
    transcript: list[tuple[str, str]],
    question: str = QUESTION,
) -> list[dict]:
    system = (
        f"{agent.persona}\n\n"
        f"You are participating in a 10-person debate on this question:\n"
        f'  "{question}"\n\n'
        "Rules:\n"
        "- Stay in character. Argue from your worldview.\n"
        "- Read the prior turns and respond to specific points other agents made — name them.\n"
        "- 3-5 sentences. No bullet lists. No hedging filler.\n"
        "- Do not prefix your response with your own name; the moderator handles that."
    )
    transcript_text = "\n\n".join(f"{name}: {text}" for name, text in transcript) or "(you speak first)"
    user = f"Debate so far:\n\n{transcript_text}\n\nNow it's your turn, {agent.name}."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3, help="number of full rounds (each agent speaks once per round)")
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5", help="OpenRouter model id")
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("error: set OPENROUTER_API_KEY", file=sys.stderr)
        return 1

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    print(f"\n=== DEBATE: {QUESTION} ===")
    print(f"model={args.model}  rounds={args.rounds}  agents={len(AGENTS)}\n")

    transcript: list[tuple[str, str]] = []
    for r in range(1, args.rounds + 1):
        print(f"\n--- Round {r} ---\n")
        for agent in AGENTS:
            messages = build_messages(agent, transcript)
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            transcript.append((agent.name, text))
            print(f"\033[1m{agent.name}:\033[0m {text}\n")
            time.sleep(0.2)

    print("\n=== END ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
