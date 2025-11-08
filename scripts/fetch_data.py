import os
import pandas as pd
from pathlib import Path
from src.data.dataset import create_csv

def main():
    # data directory
    data_dir = Path(__file__).parent.parent / "data"

    all_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in ["UGT.csv", "Activity.csv", "Substrate.csv"])
    if all_exist:
        print("All CSV files already exist. Skipping create_csv().")
    else:
        print("Some CSV files are missing. Running create_csv()...")
        print("⚠️ Please make sure you have a .env file in the project root containing:")
        print("   ACCESS_DB_PATH=/full/path/to/your/database.accdb")
        print("   ACCESS_DB_PASSWORD=yourpassword")
        create_csv()

    activity = pd.read_csv(os.path.join(data_dir, "Activity.csv"))
    UGT = pd.read_csv(os.path.join(data_dir, "UGT.csv"))
    substrate = pd.read_csv(os.path.join(data_dir, "Substrate.csv"))
    print(activity.shape)
    print(UGT.shape)
    print(substrate.shape)


if __name__ == "__main__":
    main()