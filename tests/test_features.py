import pandas as pd

from titanic_passenger_class_prediction.features import (
    add_passenger_features,
    extract_cabin_deck,
    extract_ticket_prefix,
    extract_title,
)


def make_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
            ],
            "Age": [22.0, 38.0],
            "SibSp": [1, 0],
            "Parch": [0, 0],
            "Ticket": ["A/5 21171", "113803"],
            "Fare": [7.25, 53.10],
            "Cabin": [None, "C123"],
        }
    )


def test_extract_title() -> None:
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Rothes, the Countess. of") == "Rare"
    assert extract_title(None) == "Unknown"


def test_extract_cabin_deck() -> None:
    assert extract_cabin_deck("C123") == "C"
    assert extract_cabin_deck(None) == "Unknown"


def test_extract_ticket_prefix() -> None:
    assert extract_ticket_prefix("A/5 21171") == "A_5"
    assert extract_ticket_prefix("113803") == "NONE"


def test_add_passenger_features() -> None:
    original = make_sample_dataframe()

    featured = add_passenger_features(original)

    assert featured["Title"].tolist() == ["Mr", "Mrs"]
    assert featured["FamilySize"].tolist() == [2, 1]
    assert featured["IsAlone"].tolist() == [0, 1]
    assert featured["HasCabin"].tolist() == [0, 1]
    assert featured["CabinDeck"].tolist() == ["Unknown", "C"]
    assert featured["TicketPrefix"].tolist() == ["A_5", "NONE"]

    # Confirm that the source dataframe was not modified.
    assert "Title" not in original.columns