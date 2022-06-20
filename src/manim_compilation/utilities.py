from typing import List


def bold(text: str):
    return f"\\textbf{{{text}}}"


def flatten(lst: List[List]):
    return [item for sublist in lst for item in sublist]
