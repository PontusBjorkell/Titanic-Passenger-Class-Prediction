"""Deterministic feature engineering for Titanic passenger data."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


RARE_TITLES = {
    "Capt",
    "Col",
    "Don",
    "Dona",
    "Dr",
    "Jonkheer",
    "Lady",
    "Major",
    "Rev",
    "Sir",
    "the Countess",
}

TITLE_REPLACEMENTS = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}


def extract_title(name: object) -> str:
    """
    Extract and standardize a passenger title from a full name.

    Examples
    --------
    ``Braund, Mr. Owen Harris`` becomes ``Mr``.
    """
    if not isinstance(name, str) or not name.strip():
        return "Unknown"

    match = re.search(r",\s*([^.]*)\.", name)

    if match is None:
        return "Unknown"

    title = match.group(1).strip()
    title = TITLE_REPLACEMENTS.get(title, title)

    if title in RARE_TITLES:
        return "Rare"

    return title


def extract_surname(name: object) -> str:
    """Extract a passenger surname from the full name."""
    if not isinstance(name, str) or not name.strip():
        return "Unknown"

    surname = name.split(",", maxsplit=1)[0].strip()

    return surname or "Unknown"


def extract_cabin_deck(cabin: object) -> str:
    """Extract the first cabin deck letter."""
    if not isinstance(cabin, str) or not cabin.strip():
        return "Unknown"

    first_character = cabin.strip()[0].upper()

    return first_character if first_character.isalpha() else "Unknown"


def count_cabins(cabin: object) -> int:
    """Count the number of cabin identifiers assigned to a passenger."""
    if not isinstance(cabin, str) or not cabin.strip():
        return 0

    return len(cabin.split())


def extract_ticket_prefix(ticket: object) -> str:
    """
    Extract and normalize the ticket prefix.

    The final numeric ticket identifier is removed, while numeric parts
    that belong to the prefix are retained.

    Examples
    --------
    ``A/5 21171`` becomes ``A_5``.
    ``PC 17599`` becomes ``PC``.
    ``113803`` becomes ``NONE``.
    """
    if not isinstance(ticket, str) or not ticket.strip():
        return "Unknown"

    normalized = (
        ticket.upper()
        .replace(".", "")
        .replace("/", " ")
        .strip()
    )

    parts = normalized.split()

    if len(parts) == 1 and parts[0].isdigit():
        return "NONE"

    prefix_parts = parts[:-1]

    if not prefix_parts:
        return "NONE"

    return "_".join(prefix_parts)


def add_passenger_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic passenger-level features.

    The original dataframe is not modified.
    """
    required_columns = {
        "Name",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Cannot engineer features. Missing columns: {missing}"
        )

    featured = dataframe.copy()

    featured["Title"] = featured["Name"].map(extract_title)
    featured["Surname"] = featured["Name"].map(extract_surname)

    # Family relationship counts can be missing in external datasets.
    # Treat missing counts as zero when constructing family-size features.
    sibsp = pd.to_numeric(
        featured["SibSp"],
        errors="coerce",
    ).fillna(0)

    parch = pd.to_numeric(
        featured["Parch"],
        errors="coerce",
    ).fillna(0)

    featured["FamilySize"] = (
        sibsp
        + parch
        + 1
    ).astype("int64")

    featured["IsAlone"] = (
        featured["FamilySize"]
        .eq(1)
        .astype("int8")
    )

    featured["FamilySizeGroup"] = pd.cut(
        featured["FamilySize"],
        bins=[0, 1, 4, 6, np.inf],
        labels=["Alone", "Small", "Medium", "Large"],
    )

    featured["HasCabin"] = (
        featured["Cabin"].notna().astype("int8")
    )

    featured["CabinDeck"] = featured["Cabin"].map(
        extract_cabin_deck
    )

    featured["CabinCount"] = featured["Cabin"].map(
        count_cabins
    )

    featured["TicketPrefix"] = featured["Ticket"].map(
        extract_ticket_prefix
    )

    featured["AgeGroup"] = pd.cut(
        featured["Age"],
        bins=[0, 12, 18, 35, 60, np.inf],
        labels=[
            "Child",
            "Teen",
            "Young Adult",
            "Adult",
            "Senior",
        ],
        include_lowest=True,
    )

    featured["FareLog"] = np.log1p(featured["Fare"])

    return featured