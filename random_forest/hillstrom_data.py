"""Load the Hillstrom (MineThatData 2008) email experiment and build model inputs.

Unit of observation: one customer who purchased in the 12 months before the campaign.
Each customer was randomly assigned to a Men's email, a Women's email, or no email,
and we observe whether they visited the website in the following two weeks.
"""

from pathlib import Path
import urllib.request

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = REPOSITORY_ROOT / "data" / "hillstrom_2008.csv"
# Same mirror the course starter notebook downloads from.
HILLSTROM_CSV_URL = (
    "https://raw.githubusercontent.com/AdityaDabrase/ab-testing-email-marketing/"
    "main/data/raw/hillstrom.csv"
)

EXPECTED_ROW_COUNT = 64_000
MENS_EMAIL_LABEL = "Mens E-Mail"
WOMENS_EMAIL_LABEL = "Womens E-Mail"
NO_EMAIL_LABEL = "No E-Mail"

# Pre-campaign covariates only. `history_segment` is dropped because it is just a
# binned copy of `history`, which a tree can split on directly.
NUMERIC_COVARIATES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_COVARIATES = ["zip_code", "channel"]


def load_hillstrom(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Read the raw CSV (downloading it first if missing) and fix known data quirks."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HILLSTROM_CSV_URL, csv_path)

    customers = pd.read_csv(csv_path)
    if len(customers) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} rows, found {len(customers)}")
    if customers.isna().any().any():
        raise ValueError("Unexpected missing values in the Hillstrom data")

    # The raw file misspells "Suburban".
    customers["zip_code"] = customers["zip_code"].replace({"Surburban": "Suburban"})
    return customers


def email_recipients_only(customers: pd.DataFrame) -> pd.DataFrame:
    """Keep the two email arms and add a 0/1 treatment indicator (1 = Men's email).

    Our question compares the two emails against each other, so the no-email arm is
    not needed. Because assignment was random, the email recipients are still a
    representative sample of the full customer base.
    """
    is_email_arm = customers["segment"].isin([MENS_EMAIL_LABEL, WOMENS_EMAIL_LABEL])
    email_recipients = customers.loc[is_email_arm].reset_index(drop=True)
    email_recipients["received_mens_email"] = (
        email_recipients["segment"] == MENS_EMAIL_LABEL
    ).astype(int)
    return email_recipients


def build_covariate_matrix(customers: pd.DataFrame) -> pd.DataFrame:
    """Numeric covariates plus one-hot encoded zip_code and channel (all levels kept;
    random forests do not need a reference level)."""
    one_hot_categoricals = pd.get_dummies(
        customers[CATEGORICAL_COVARIATES], prefix=CATEGORICAL_COVARIATES, dtype=int
    )
    return pd.concat([customers[NUMERIC_COVARIATES], one_hot_categoricals], axis=1)
