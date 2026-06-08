from tp_ia.data.dataset import Dataset
from tp_ia.training import DEFAULT_MODELS, Experiment


def main():
    dataset = Dataset("breast_cancer")
    print(dataset)
    print()

    experiment = Experiment(dataset, DEFAULT_MODELS)
    experiment.run()
    print(experiment)


if __name__ == "__main__":
    main()
