from abc import ABC, abstractmethod
import pandas as pd

from us_visa.logger import logging
from us_visa.exception import CustomException

# Abstract Base Class for Missing Value Handling Strategy.
# -------------------------------------------------
# Subclasses must implemet the handle method
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass

# Concrete Strategy for Dropping Missing Values
# --------------------------------------------
class DropMissingValues(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
       """
       Initializes the DropMissingValues with specific parameters.

       Parameters:
       axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
       thresh (int): Require that many non-NA values. Rows/Columns with less than thresh non-NA values are dropped.
       """
       self.axis = axis
       self.thresh = thresh
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned
        
# Concrete Strategy class for Filling Missing Values
# -------------------------------------------
class FillMissingValues(MissingValueHandlingStrategy):
    def __init__(self, method=None, fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
        fill_value (any): The constant value to fill missing values when method='constant'.
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified method or constant value.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled.
        """
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            # Fill numeric columns with mean
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())
        elif self.method == 'median':
            # Fill numeric columns with median
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].median())
        elif self.method == "mode":
            # Fill non-numeric columns with mode
            category_columns = df_cleaned.select_dtypes(exclude="number").columns
            df_cleaned[category_columns] = df_cleaned[category_columns].fillna(df[category_columns].mode().iloc[0])
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned

# Context Class for Handling Missing Values
# ----------------------------------------
class MissingValueHandler:
    def __init__(self):
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = None

    def handle_missing_values(self, strategy_class, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Sets and executes the specified missing value handling strategy.
        
        Parameters:
        strategy_class (class): The strategy class to instantiate and use.
        df (pd.DataFrame): The input DataFrame containing missing values.
        kwargs: Additional arguments for the strategy (e.g., axis for dropping, method for filling).
        
        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy_class(**kwargs)
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)




# Usage example
# if __name__ == "__main__":
#     # Sample DataFrame with missing values for demonstration
#     data = {'A': [1, None, 3, None, 5], 'B': [None, 'b', 'c', None, 'e']}
#     df = pd.DataFrame(data)

#     # Initialize MissingValueHandler
#     missing_value_handler = MissingValueHandler()

#     # Drop rows with missing values
#     df_dropped = missing_value_handler.handle_missing_values(DropMissingValues, df, axis=0, thresh=2)
#     print("\nDataFrame after dropping rows with missing values:")
#     print(df_dropped)

#     # Fill missing values with mean for numeric columns
#     df_filled_mean = missing_value_handler.handle_missing_values(FillMissingValues, df, method="mode")
#     print("\nDataFrame after filling missing values with mean:")
#     print(df_filled_mean)

#     # Fill missing values with a constant value
#     df_filled_constant = missing_value_handler.handle_missing_values(FillMissingValues, df, method="constant", fill_value=0)
#     print("\nDataFrame after filling missing values with a constant value (0):")
#     print(df_filled_constant)