import os
from getpass import getpass
from semantic_router.encoders import HuggingFaceEncoder


from datasets import load_dataset

dataset = load_dataset("jamescalam/ai-arxiv2", split="train")
dataset

encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

from semantic_router.splitters import RollingWindowSplitter
from semantic_router.utils.logger import logger

logger.setLevel("WARNING")  # reduce logs from splitter

splitter = RollingWindowSplitter(
    encoder=encoder,
    dynamic_threshold=True,
    min_split_tokens=100,
    max_split_tokens=500,
    window_size=2,
    plot_splits=True,  # set this to true to visualize chunking
    enable_statistics=True  # to print chunking stats
)

splits = splitter([dataset["content"][2]])
splitter.print(splits[:3])