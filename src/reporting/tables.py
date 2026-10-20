"""Small helpers for markdown tables."""


def markdown_table(headers: list[str], rows: list[list]) -> list[str]:
    """Return the lines of a markdown table; columns after the first are right-aligned."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return lines


def mean_sd(mean: float, sd: float) -> str:
    """Format as 'mean ± sd' with three decimals."""
    return f"{mean:.3f} ± {sd:.3f}"
