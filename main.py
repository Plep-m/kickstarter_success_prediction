from tp_ia.data.loader import load_dataset


def main():
    X, _ = load_dataset("iris")
    print(f"Shape: {X.shape}")


if __name__ == "__main__":
    main()
