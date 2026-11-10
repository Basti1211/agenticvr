"""Data-layer naming adapter for retrieval-loop queries."""


def retrieval_dataset_name(dataset: str) -> str:
    """Translate public dataset names to the data-layer query namespace."""

    return "vbslhe" if dataset == "lhe" else dataset
