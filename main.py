import mlflow
import os
import hydra
from omegaconf import DictConfig

# This automatically reads in the configuration
@hydra.main(config_name='config')
def process_args(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = list(config["main"]["execute_steps"])

    # Download step
    if "fetch_data" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "fetch_data"),
            "main",
            parameters={
                "artifact_name": "raw_data.csv",
                "url": config["data"]["file_url"],
                "dataset": config["data"]["dataset"],
            }
        )

    if "preprocessing" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "preprocessing"),
            "main",
            parameters={
                "input_artifact": "raw_data.csv:latest",
                "features": "clean_features:v2",
                "target": "labels:v1"
            }
        )

    if "data_segregation" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "data_segregation"),
            "main",
            parameters={
                "artifact_name_feature": "clean_features:v2",
                "artifact_name_target": "labels:v1",
                "train_feature_artifact": "train_x",
                "train_target_artifact": "train_y",
                "val_feature_artifact": "val_x",
                "val_target_artifact": "val_y",
                "test_feature_artifact": "test_x",
                "test_target_artifact": "test_y",
                "test_size": config["data"]["test_size"],
                "val_size": config["data"]["val_size"],
                "seed": config["main"]["random_seed"]
            }
        )

    if "train" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "train"),
            "main",
            parameters={
                "train_feature_artifact": "train_x:latest",
                "train_target_artifact": "train_y:latest",
                "val_feature_artifact": "val_x:latest",
                "val_target_artifact": "val_y:latest",
                "encoder": "target_encoder",
                "inference_model": "vit_l_32.pth",
                "batch_size": config["model"]["vision_transformer"]["batch_size"],
                "seed": config["main"]["random_seed"],
                "epochs": config["model"]["vision_transformer"]["epochs"],
                "learning_rate": config["model"]["vision_transformer"]["learning_rate"]
            }
        )

    if "test" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "test"),
            "main",
            parameters={
                "test_feature_artifact": "test_x:latest",
                "test_target_artifact": "test_y:latest",
                "encoder": "target_encoder:latest",
                "inference_model": "vit_l_32.pth:latest",
                "batch_size": config["model"]["vision_transformer"]["batch_size"],
            }
        )


if __name__ == "__main__":
    process_args()