"""Prompt builders for survey response generation."""

from __future__ import annotations

from typing import Dict, List


def bigfive_direct(question: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an AI assistant. Respond to a Big Five Personality Test statement "
        "by selecting one option from the scale below.\n\n"
        "1 = Strongly disagree\n"
        "2 = Disagree\n"
        "3 = Neutral\n"
        "4 = Agree\n"
        "5 = Strongly agree\n\n"
        "Respond with only a single number from 1 to 5."
    )
    user_prompt = (
        f'Please read the following statement:\n\n"{question}"\n\n'
        "To what extent do you agree? Answer with the number only."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def bigfive_with_persona(question: str, persona: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are roleplaying as a real human participant described below.\n\n"
        f"{persona}\n\n"
        "You are taking the Big Five Personality Test. For each statement, respond "
        "as this person by selecting one number from 1 to 5.\n\n"
        "1 = Strongly disagree\n"
        "2 = Disagree\n"
        "3 = Neutral\n"
        "4 = Agree\n"
        "5 = Strongly agree\n\n"
        "Respond with only a single number from 1 to 5."
    )
    user_prompt = (
        "How much would this person agree with the following statement?\n\n"
        f'"{question}"\n\n'
        "Answer with a number only."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def ood_with_persona(item: Dict, persona: str) -> List[Dict[str, str]]:
    system_prompt = item["prompt_with_persona_system"].replace("{ROLE_PROFILE}", persona)
    user_prompt = item["prompt_with_persona_user"].replace("{ROLE_PROFILE}", persona)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
