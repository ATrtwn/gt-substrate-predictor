from src.data.preprocessing import prepare_dataset
from src.data.data_split import split_dataset
from src.data.embeddings import create_embeddings

def main():
    print("1/3 Preprocessing dataset...")
    df_all = prepare_dataset()

    print("2/3 Splitting into C1/C2/C3...")
    df = split_dataset(df_all)

    print("3/3 Generating embeddings...")
    create_embeddings()

if __name__ == "__main__":
    main()