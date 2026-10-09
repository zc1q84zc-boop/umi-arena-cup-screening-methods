"""Exact UMI Arena cup primitives; instructions, not an action controller.

Source checked 2026-10-08: https://umi-arena.airoa.io/evaluation
The evaluator changes the instruction only after a primitive succeeds.
"""

PROMPT_SOURCE = "https://umi-arena.airoa.io/evaluation"
PROMPT_PROTOCOL = "umi_arena_cup_primitives_20261008"
RIGHT_PLACE = "Pick up the cup with your right hand and set it on the plate"
LEFT_RETURN = "Pick up the cup with your left hand and return the cup to its original position"


def cup_instruction(stage):
    if stage == "place_on_plate":
        return RIGHT_PLACE
    if stage == "return_to_origin":
        return LEFT_RETURN
    raise ValueError("no active official cup primitive for task stage")


def validate_instruction(prompt):
    if not isinstance(prompt, str) or prompt not in (RIGHT_PLACE, LEFT_RETURN):
        raise ValueError("prompt must be an exact official cup primitive")
    return prompt
