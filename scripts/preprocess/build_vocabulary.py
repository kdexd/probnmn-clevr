import argparse
import json
import os

from loguru import logger
from typing import List, Set
from mypy_extensions import TypedDict


parser = argparse.ArgumentParser(
    description="""
    Build a vocabulary out of CLEVR v1.0 annotation json file. This vocabulary would be something
    which AllenNLP can understand. It's a directory with separate text files containing unique
    tokens of questions, programs and answers.
    """,
    epilog="""
    Vocabulary of synthetic questions and programs contains a limited number of unique tokens,
    so we include all the tokens without setting an inclusion threshold.
    """,
)

parser.add_argument(
    "-c",
    "--clevr-jsonpath",
    default="data/CLEVR_train_questions.json",
    help="Path to CLEVR v1.0 train annotation json file.",
)
parser.add_argument(
    "-o",
    "--output-dirpath",
    default="data/clevr_vocabulary",
    help="Path to a (non-existent directory to save the vocabulary.",
)


# ------------------------------------------------------------------------------------------------
# All the punctuations in CLEVR question sequences.
PUNCTUATIONS: List[str] = ["?", ".", ",", ";"]

# Special tokens which should be added (all, or a subset) to the vocabulary.
SPECIAL_TOKENS: List[str] = ["@@PADDING@@", "@@UNKNOWN@@", "@start@", "@end@"]

# Type for a single token of program sequence in each CLEVR example annotation.
ProgramToken = TypedDict(
    "ProgramToken", {"inputs": List[int], "function": str, "value_inputs": List[str]}
)

# Type for each CLEVR example annotation.
ClevrExample = TypedDict(
    "ClevrExample",
    {
        "question": str,
        "program": List[ProgramToken],
        "answer": str,
        "image_index": int,
        "image_filename": str,
        "question_index": int,
        "question_family_index": int,
        "split": str,
    },
)
# ------------------------------------------------------------------------------------------------


def build_question_vocabulary(clevr_json: List[ClevrExample]) -> List[str]:
    """Given a list of CLEVR example annotations, return a list of unique question tokens."""

    question_tokens: Set[str] = set()
    # Accumulate unique question tokens from all question sequences.
    for item in clevr_json:
        sequence: str = item["question"]
        # Add a leading space before punctuations to make sure they don't mix with word tokens.
        for punctuation in PUNCTUATIONS:
            sequence = sequence.replace(punctuation, f" {punctuation}")

        sequence_tokens = [token for token in sequence.split(" ") if token not in {"?", "."}]
        question_tokens = question_tokens.union(set(sequence_tokens))

    question_vocabulary: List[str] = sorted(list(question_tokens))
    return question_vocabulary


def build_program_vocabulary(clevr_json: List[ClevrExample]) -> List[str]:
    """Given a list of CLEVR example annotations, return a list of unique program tokens."""

    program_tokens: Set[str] = set()
    # Accumulate unique question tokens from all question sequences.
    for item in clevr_json:
        sequence: List[ProgramToken] = item["program"]

        for sequence_element in sequence:
            # For example: "scene", "count", "filter_size", etc.
            program_token = sequence_element["function"]

            # For example: {"inputs": [0], "function": "filter_size", "value_inputs": "large"}
            # Make the program token - "filter_size[large]"
            if len(sequence_element["value_inputs"]) > 0:
                program_token = program_token + "[" + sequence_element["value_inputs"][0] + "]"

            program_tokens.add(program_token)

    program_vocabulary: List[str] = sorted(list(program_tokens))
    return program_vocabulary


if __name__ == "__main__":

    args = parser.parse_args()
    logger.info(f"Loading annotations json from {args.clevr_jsonpath}...")
    clevr_json = json.load(open(args.clevr_jsonpath))["questions"]

    logger.info("Building question vocabulary...")
    question_vocabulary: List[str] = build_question_vocabulary(clevr_json)
    question_vocabulary = SPECIAL_TOKENS + question_vocabulary
    logger.info(f"Question vocabulary size (with special tokens): {len(question_vocabulary)}")

    logger.info("Building program vocabulary...")
    program_vocabulary: List[str] = build_program_vocabulary(clevr_json)
    program_vocabulary = SPECIAL_TOKENS + program_vocabulary
    logger.info(f"Program vocabulary size (with special tokens): {len(program_vocabulary)}")

    logger.info("Building answer vocabulary...")
    # Only @@UNKNOWN@@ for answer vocabulary, because answers are not a "sequence".
    answer_vocabulary: List[str] = sorted(list(set([item["answer"] for item in clevr_json])))
    answer_vocabulary = answer_vocabulary + ["@@UNKNOWN@@"]
    logger.info(f"Answer vocabulary size: {len(answer_vocabulary)}")

    # Write the vocabulary to separate namespace files in directory.
    logger.info(f"Writing the vocabulary to {args.output_dirpath}...")
    logger.info("Namespaces: programs, questions, answers.")
    logger.info("Non-padded namespaces: answers.")

    os.makedirs(args.output_dirpath, exist_ok=True)

    # DO NOT write @@PADDING@@ token, AllenNLP would always add it internally.
    with open(os.path.join(args.output_dirpath, "questions.txt"), "w") as f:
        for question_token in question_vocabulary[1:]:
            f.write(question_token + "\n")

    with open(os.path.join(args.output_dirpath, "programs.txt"), "w") as f:
        for program_token in program_vocabulary[1:]:
            f.write(program_token + "\n")

    with open(os.path.join(args.output_dirpath, "answers.txt"), "w") as f:
        for answer_token in answer_vocabulary:
            f.write(answer_token + "\n")

    with open(os.path.join(args.output_dirpath, "non_padded_namespaces.txt"), "w") as f:
        f.write("answers")
