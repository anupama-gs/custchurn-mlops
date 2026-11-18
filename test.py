import mlflow
from mlflow.tracking import MlflowClient


def main():
    # Copy your failing script's MLflow URI settings here.
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Registry URI:", mlflow.get_registry_uri())

    client = MlflowClient()
    for model in client.search_registered_models():
        print(repr(model.name))


if __name__ == "__main__":
    main()