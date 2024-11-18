import os
from pathlib import Path
import zipfile
from abc import ABC, abstractmethod
import pandas as pd
from us_visa.logger import logging
from us_visa.exception import CustomException

# Define an abstract class for Data Ingestor/ Interface
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, path_to_data: Path) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass

PATH_TO_EXTRACTED_DATA = r"us_visa/data/extracted_data"

# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    logging.info("Enter in ZipDataIngestor class")

    def ingest(self, file_path: Path) -> pd.DataFrame:
        """Extract a .zip file and returns the content as a pandas Dataframe."""
        # Ensure the file is .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")
        
        # Extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            logging.info("zipfile extracted and stored at us_visa/data/extracted_data directory")
            zip_ref.extractall(PATH_TO_EXTRACTED_DATA)

        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir(PATH_TO_EXTRACTED_DATA)
        csv_files = [file for file in extracted_files if file.endswith('.csv')]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV file found. Please specify which one to use.")
        
        # Read the CSV file into a DataFrame
        csv_file_path = os.path.join(PATH_TO_EXTRACTED_DATA, csv_files[0])
        df = pd.read_csv(csv_file_path)
        logging.info("ZIP file is extracted into CSV file")
        logging.info("Exit from ZipDataIngestor class")

        # Return the DataFrame
        return df

class JsonDataIngestor(DataIngestor):
    logging.info("Enter in JsonDataIngestor class")
    def ingest(self, file_path: Path) -> pd.DataFrame:
        pass

        
# Implement a Factory to Create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        "Returns the appropriate DataIngestor based on file extension."
        if file_extension == '.zip':
            return ZipDataIngestor()
        elif file_extension == '.json':
            return JsonDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")

# Example usage:
if __name__ == "__main__":
    # Specify the file path
    file_path = "/Users/aadarsh/Desktop/Data Scientist/Projects/US-Visa-Approval-Prediction/us_visa/data/archive.zip"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension) 

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)

    # Now df contains the DataFrame from the exctracted csv
    print(df.head()) # Display the first 5 rows of the DataFrame
