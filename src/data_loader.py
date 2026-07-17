import pandas as pd
from pathlib import Path


def load_stock_data(data_dir):
    """
    Load all stock files from the dataset.

    Parameters
    ----------
    data_dir : str or Path
        Path to the Stocks folder.

    Returns
    -------
    dict
        Dictionary containing stock DataFrames.
    """

    stock_data = {}

    for file in Path(data_dir).glob("*.txt"):

        try:

            df = pd.read_csv(file)

            if df.empty:
                continue

            stock_name = file.stem.replace(".us", "")

            stock_data[stock_name] = df

        except Exception:
            continue

    return stock_data
def load_single_stock(data_dir, ticker):
    """
    Load only one stock.
    """

    file_path = Path(data_dir) / f"{ticker.lower()}.us.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"{ticker} not found.")

    df = pd.read_csv(file_path)

    return df