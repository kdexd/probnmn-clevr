import argparse
import json

from typing import Any, Dict, List
from mypy_extensions import TypedDict

from allennlp.data import Vocabulary
import h5py
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="""
    Given CLEVR v1.0 annotations json, tokenize all programs, questions and answers, and save them
    along with corresponding image indices to an H5 file.
    """
)

parser.add_argument(
    "-c",
    "--clevr-jsonpath",
    default="data/CLEVR_train_questions.json",
    help="Path to CLEVR v1.0 train / val annotation json file.",
)
parser.add_argument(
    "-v",
    "--vocab-dirpath",
    default="data/vocabulary",
    help="Path to a directory containing AllenNLP compatible vocabulary.",
)
parser.add_argument(
    "-o",
    "--output-h5path",
    default="data/clevr_train_tokens.h5",
    help="Path to save tokenized components in an H5 file.",
)
parser.add_argument("-s", "--split", default="train", choices=["train", "val", "test"])

# ------------------------------------------------------------------------------------------------
# All the punctuations in CLEVR question sequences.
PUNCTUATIONS: List[str] = ["?", ".", ",", ";"]

# Type for a single token of program sequence in each CLEVR example annotation.
ProgramToken = TypedDict(
    "ProgramToken", {"inputs": List[int], "function": str, "value_inputs": List[str]}
)
# ------------------------------------------------------------------------------------------------


def tokenize_program(program_list: List[ProgramToken]) -> List[str]:
    """Given a program list from CLEVR annotations json, tokenize it in prefix notation."""
    program_prefix: List[str] = []

    def build_subtree(program_token: ProgramToken) -> Dict[str, Any]:
        function: str = program_token["function"]
        if len(program_token["value_inputs"]) > 0:
            function += "[" + ",".join(program_token["value_inputs"]) + "]"
        return {
            "function": function,
            "inputs": [build_subtree(program_list[i]) for i in program_token["inputs"]],
        }

    # Recursive structure, a tree with last program token as root node.
    # Prefix notation of a program is pre-order traversal of this tree.
    program_tree: Dict[str, Any] = build_subtree(program_list[-1])

    def pre_order_traversal(program_tree: Dict[str, Any]):
        program_prefix.append(program_tree["function"])
        for i in range(len(program_tree["inputs"])):
            pre_order_traversal(program_tree["inputs"][i])

    pre_order_traversal(program_tree)
    return program_prefix


def tokenize_question(question: str) -> List[str]:
    """Given a question from CLEVR annotations json, tokenize it as words."""
    for punctuation in PUNCTUATIONS:
        question = question.replace(punctuation, f" {punctuation}")

    question_tokens = [token for token in question.split(" ") if token not in {"?", ".", ""}]
    return question_tokens


if __name__ == "__main__":

    args = parser.parse_args()
    print(f"Loading annotations json from {args.clevr_jsonpath}...")
    clevr_json = json.load(open(args.clevr_jsonpath))["questions"]

    vocabulary = Vocabulary.from_files(args.vocab_dirpath)

    # Collect image indices and answers corresponding to question-program pairs.
    image_indices: List[int] = []
    answers: List[int] = []

    # Tokenize and pad all questions and programs up to maximum length.
    tokenized_questions: List[List[str]] = []
    tokenized_programs: List[List[str]] = []

    print("Tokenizing questions, programs and answers...")
    for item in tqdm(clevr_json):
        tokenized_questions.append(tokenize_question(item["question"]))
        image_indices.append(item["image_index"])
        if args.split != "test":
            tokenized_programs.append(tokenize_program(item["program"]))
            answers.append(vocabulary.get_token_index(item["answer"], namespace="answers"))

    question_max_length: int = max([len(q) for q in tokenized_questions])

    if args.split != "test":
        program_max_length: int = max([len(p) for p in tokenized_programs])

    print(f"Saving tokenized data to {args.output_h5path}...")

    output_h5 = h5py.File(args.output_h5path)
    output_h5["image_indices"] = image_indices

    output_h5.create_dataset(
        "questions", (len(tokenized_questions), question_max_length), dtype=int
    )
    for i, tokenized_question in enumerate(tqdm(tokenized_questions, desc="questions")):
        output_h5["questions"][i, : len(tokenized_question)] = [
            vocabulary.get_token_index(q, namespace="questions") for q in tokenized_question
        ]

    if args.split != "test":
        output_h5["answers"] = answers

        output_h5.create_dataset(
            "programs", (len(tokenized_programs), program_max_length), dtype=int
        )
        for i, tokenized_program in enumerate(tqdm(tokenized_programs, desc="programs")):
            output_h5["programs"][i, : len(tokenized_program)] = [
                vocabulary.get_token_index(p, namespace="programs") for p in tokenized_program
            ]

    output_h5.attrs["split"] = args.split
    output_h5.close()
