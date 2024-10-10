from clearbg import logger
from clearbg.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Main Pipeline"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e