from tp_ia.data.dataset import Dataset


def main():
    dataset = Dataset("breast_cancer", test_fraction=0.15, validation_fraction=0.15)

    print("--- Complete Dataset Overview ---")
    print(dataset)

    print("--- Splitting Verification ---")
    print(f"Train features shape:      {dataset.training.features.shape}")
    print(f"Validation features shape: {dataset.validation.features.shape}")
    print(f"Test features shape:       {dataset.testing.features.shape}")


if __name__ == "__main__":
    main()
