import os
import glob
import mlflow
import mlflow.pytorch
import torch
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import transforms
from clearbg.model.u2net import U2NET  # Update with your model import
from clearbg.utils.utils import SalObjDataset, RescaleT, ToTensorLab  # Update with your data loader imports
from clearbg.utils.common import read_yaml, save_json  # Update with your utility imports
import numpy as np
from PIL import Image

# SET YOUR MLFLOW TRACKING URI HERE
mlflow.set_tracking_uri("")

@dataclass(frozen=True)
class EvaluationConfig:
    model_path: Path
    test_data_dir: Path
    prediction_dir: Path
    params: dict

class ConfigurationManager:
    def __init__(self, config_filepath="config.yaml", params_filepath="params.yaml"):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
    
    def get_evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            model_path=Path(self.config.training.root_dir) / "saved_models" / "u2net" / "u2net.pth",
            test_data_dir=Path(self.config.data_ingestion.root_dir) / "test_data" / "test_images",
            prediction_dir=Path(self.config.data_ingestion.root_dir) / "test_data" / "u2net_results",
            params=self.params
        )

class MlflowEval:
    def __init__(self, config_filepath="config.yaml", params_filepath="params.yaml"):
        self.config_manager = ConfigurationManager(config_filepath, params_filepath)
        self.eval_config = self.config_manager.get_evaluation_config()

    def load_model(self, model_path: Path) -> torch.nn.Module:
        model = U2NET(3, 1)  # Ensure this matches your model's input/output
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model

    def evaluate_model(self, model: torch.nn.Module, dataloader: DataLoader) -> float:
        total_loss = 0
        num_samples = len(dataloader)
        
        # Assuming you have a loss function defined elsewhere
        criterion = torch.nn.BCELoss()

        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data['image'], data['label']
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
        average_loss = total_loss / num_samples
        return average_loss

    def save_output(self, predictions, img_name_list, prediction_dir):
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        
        for img_name, pred in zip(img_name_list, predictions):
            pred_np = pred.squeeze().cpu().numpy()
            im = Image.fromarray((pred_np * 255).astype(np.uint8)).convert('RGB')
            img_name_only = os.path.basename(img_name).split('.')[0]
            im.save(os.path.join(prediction_dir, f"{img_name_only}.png"))

    def run_evaluation(self):
        image_list = glob.glob(os.path.join(self.eval_config.test_data_dir, '*'))
        
        test_dataset = SalObjDataset(
            img_name_list=image_list,
            lbl_name_list=[],  # Assuming no labels for test set
            transform=transforms.Compose([
                RescaleT(320),
                ToTensorLab(flag=0)
            ])
        )
        
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Load model and evaluate
        model = self.load_model(self.eval_config.model_path)

        with mlflow.start_run():
            mlflow.log_params(self.eval_config.params)

            average_loss = self.evaluate_model(model, test_dataloader)
            mlflow.log_metrics({"average_loss": average_loss})

            # Save outputs if needed
            predictions = []
            for data in test_dataloader:
                inputs = data['image']
                with torch.no_grad():
                    output = model(inputs)
                    predictions.append(output)

            self.save_output(predictions, image_list, self.eval_config.prediction_dir)

if __name__ == "__main__":
    evaluator = MlflowEval()
    evaluator.run_evaluation()
