import os
import requests
import zipfile
from clearbg import logger
from clearbg.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_train_data(self) -> str:
        """
        Download Train Dataset
        """
        try:
            train_source_url = self.config.train_source_url
            logger.info(f"Downloading TRAIN DATASET from: {train_source_url}")
            
            response = requests.get(train_source_url)
            response.raise_for_status() 

            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)
            filename = os.path.basename(train_source_url)
            file_path = os.path.join(root_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded TRAIN DATASET: {file_path}")
            return file_path 
            
        except Exception as e:
            logger.error(f"Failed to download train dataset: {e}")
            raise e
    
    def download_test_data(self) -> str:
        """
        Download Test Dataset
        """
        try:
            test_source_url = self.config.test_source_url
            logger.info(f"Downloading TEST DATASET from: {test_source_url}")

            response = requests.get(test_source_url)
            response.raise_for_status()

            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)
            filename = os.path.basename(test_source_url)
            file_path = os.path.join(root_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded TEST DATASET: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to download test dataset: {e}")
            raise e
    
    def download_data(self, mode="all") -> str:
        """
        Download Train and Test Datasets
        """
        try:
            if mode == "train":
                return self.download_train_data()
            elif mode == "test":
                return self.download_test_data()
            elif mode == "all":
                train_path = self.download_train_data()
                test_path = self.download_test_data()

        except Exception as e:
            logger.error(f"Failed to download datasets: {e}")
            raise e 

    def extract_train_data(self) -> str:
        """
        Extract Zipped Train Dataset
        """
        try:
            root_dir = self.config.root_dir
            train_local_zipped_path = self.config.train_local_zipped_path
            
            os.makedirs(root_dir, exist_ok=True)

            logger.info(f"Extracting TRAIN DATASET from: {train_local_zipped_path}")
            with zipfile.ZipFile(train_local_zipped_path, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
            logger.info(f"Extracted TRAIN DATASET to: {root_dir}")
            

        except Exception as e:
            logger.error(f"Failed to extract train dataset: {e}")
            raise e

    def extract_test_data(self) -> str:
        """
        Extract Zipped Test Dataset
        """
        try:
            root_dir = self.config.root_dir
            test_local_zipped_path = self.config.test_local_zipped_path
            
            os.makedirs(root_dir, exist_ok=True)

            logger.info(f"Extracting TEST DATASET from: {test_local_zipped_path}")
            with zipfile.ZipFile(test_local_zipped_path, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
            logger.info(f"Extracted TEST DATASET to: {root_dir}")
            
        except Exception as e:
            logger.error(f"Failed to extract test dataset: {e}")
            raise e

    def extract_data(self, mode="all") -> str:
        """
        Extract Train and Test Datasets
        """
        try:
            if mode == "train":
                return self.extract_train_data()
            elif mode == "test":
                return self.extract_test_data()
            elif mode == "all":
                train_path = self.extract_train_data()
                test_path = self.extract_test_data()

        except Exception as e:
            logger.error(f"Failed to extract datasets: {e}")
            raise e