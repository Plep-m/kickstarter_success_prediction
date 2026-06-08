from tp_ia.data import Dataset


def main():
    dataset = Dataset("breast_cancer")

    X, y = dataset.X, dataset.y

    print(dataset)


if __name__ == "__main__":
    main()
