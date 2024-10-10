from clearbg.components.model_eval_mlflow import MlflowEval
from clearbg import logger

STAGE_NAME = "MLFlow Model Evaluation"

class ModelEvalPipeline:

    def __init__(self):
        pass

    def main(self):
        model_eval = MlflowEval()
        model_eval.run_evaluation()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvalPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 