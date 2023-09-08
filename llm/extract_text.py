from typing import Generator, Tuple, Iterator
from pathlib import Path
import logging
import json
from tqdm import tqdm

import odf
from odf import text
from odf.opendocument import load

import sys

sys.path.append("/content/llm-tolkien")

from llm import config


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def extract_odt(file_path: Path) -> str:
    """Extract text from an ODT file.

    Args:
        file_path (Path): Path to the ODT file.

    Returns:
        str: Extracted text from the ODT file.
    """
    LOGGER.info(f"Extracting text from {file_path}")
    text_content = []
    doc = load(file_path)
    for para in doc.getElementsByType(text.P):
        if para.firstChild is not None:
            if para.firstChild.__class__.__name__ == "Text":
                text_content.append(para.firstChild.data)

    return "\n".join(text_content)


def to_jsonl(text_content: str, path: Path) -> None:
    LOGGER.info(f"Start writing to {path}")
    # Print the extracted text before writing it to the JSONL file
    print(text_content)

    # We append text to the existing file with "a" mode (append)
    with open(path, 'a') as f:
        dict_page = {"1": text_content}
        json.dump(dict_page, f)
        f.write('\n')
    LOGGER.info(f"Finished writing to {path}")



if __name__ == "__main__":
    file_path = "/content/llm-tolkien/llm/originalpdf/Test.odt"
    text_content = extract_odt(Path(file_path))
    to_jsonl(text_content, config.extraction_path)
