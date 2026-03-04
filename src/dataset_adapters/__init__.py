# src/dataset_adapters/__init__.py

from .amazon_reviews import AmazonReviewsAdapter
from .kuairec import KuaiRecAdapter


def load_adapter(name):

    if name == "amazon_reviews":
        return AmazonReviewsAdapter()

    if name == "kuairec":
        return KuaiRecAdapter()

    raise ValueError(f"Unknown adapter {name}")