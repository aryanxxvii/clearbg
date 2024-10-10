from clearbg import logger
from clearbg.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from clearbg.pipeline.stage_02_model_trainer import ModelTrainingPipeline


STAGE_NAME = "Data Ingestion Pipeline"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    model_trainer = ModelTrainingPipeline()
    model_trainer.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e