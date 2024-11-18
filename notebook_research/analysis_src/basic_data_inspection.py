from abc import ABC, abstractmethod
import pandas as pd

from us_visa.logger import logging
from us_visa.exception import CustomException


# Abstract Base Class for Data Inspection Strategies | Strategy Class
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implemet the inspect method
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Return:
        None: This method prints the inspection results directly.
        """
        pass


# Concrete Strategy Class for Displaying rows of dataframe
# -----------------------------------------------------------------
# This strategy inspect the first n rows or default to 5 rows of dataframe
class HeadInspection(DataInspectionStrategy):
    def __init__(self, n:int = None):
         """
         Initializes the strategy with the number of rows to display.
            
         Parameters: n (int, optional): Number of rows to display from the top of the DataFrame. 
         Defaults to None, in which case the default number of rows (5) is shown.
         """
         self.n = n
    
    def inspect(self, df: pd.DataFrame):
        """
        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the given number of rows otherwise default 5 rows to the console.
        """
        if self.n is None:
            print("\nDefault First 5 rows")
            print(df.head())  # Displays the default 5 rows
        else:
            print(f"\nFirst {self.n} Rows:") 
            print(df.head(self.n)) # Displays the first n rows


# Concrete Strategy Class for Displaying rows of dataframe
# -----------------------------------------------------------------
# This strategy inspect the last n rows or default to 5 rows of dataframe
class TailInspection(DataInspectionStrategy):
    def __init__(self, n:int = None):
         """
         Initializes the strategy with the number of rows to display.
            
         Parameters: n (int, optional): Number of rows to display from the last of the DataFrame. 
         Defaults to None, in which case the default number of rows (5) is shown.
         """
         self.n = n
    
    def inspect(self, df: pd.DataFrame):
        """
        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the given number of rows otherwise default 5 rows to the console.
        """
        if self.n is None:
            print("\nDefault Last 5 rows")
            print(df.tail()) # Displays the default 5 rows
        else:
            print(f"\nLast {self.n} Rows:") 
            print(df.tail(self.n)) # Displays the last n rows


# Concrete Strategy Class for Displaying Dimensions of dataframe
# -----------------------------------------------------------------
# This strategy inspect the shape of the dataframe
class ShapeInspection(DataInspectionStrategy):
    
    def inspect(self, df: pd.DataFrame):

        """
        Inspects and print the rows and columns that dataframe has. Also known as Dimension
        
        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the dimension of DataFrame
        """
        print(f"\nRows and Columns in dataset")
        print(df.shape)


# Concrete Strategy Class for Data Types Inspection
# -------------------------------------------------
# This strategy inspects the data types of each column and counts non-null values.
class DataTypesInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.
        
        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())


# Concrete Strategy class for Summary Statistics Inspection
# -----------------------------------------------------
# This strategy provides summary statistics for both numerical and categorical features.
class SummaryStatisticsInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


# Concrete Strategy class for Numerical Features
# ----------------------------------------------
# This strategy inspect number of numerical feature in DataFrame
class NumericalFeatureInspection(DataInspectionStrategy):
    """
    Prints and returns the numerical features present in DataFrame

    Parameters:
    df (pd.DataFrame): The dataframe to be inspected

    Returns:
    numerical_features: prints List of numerical features in DataFrame
    """ 
    def inspect(self, df: pd.DataFrame):
        numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
        print(f"This dataset has {len(numerical_features)} Numerical Features and they are : {numerical_features}")
        return numerical_features
    


# Concrete Strategy class for Categorical Features
# ----------------------------------------------
# This strategy inspect number of categorical feature in DataFrame
class CategoricalFeatureInspection(DataInspectionStrategy):
    """
    Prints and returns the categorical features present in DataFrame

    Parameters:
    df (pd.DataFrame): The dataframe to be inspected

    Returns:
    categorical_features: List of categorical features in DataFrame
    """ 
    def inspect(self, df: pd.DataFrame):
        categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
        print(f"This dataset has {len(categorical_features)} Categorical Features and they are : {categorical_features}")
        return categorical_features


# Concrete Strategy class for Proportion of count data on Categorical Features
# ----------------------------------------------
# This strategy inspect number of categorical feature in DataFrame
class ProportionFeatureInspection(DataInspectionStrategy):
    """
    Prints the Proportion of count data on categorical column

    Parameters:
    df (pd.DataFrame): The dataframe to be inspected

    Returns:
    None: Prints the proportion of each categorical feature present in dataframe
    """
    def inspect(self, df: pd.DataFrame):
        categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
        for feature in categorical_features:
            print(df[feature].value_counts(normalize=True)*100)
            print("-------------------------------------------")


# Concrete Strategy class for number of count of value on Categorical Features
# ----------------------------------------------
# This strategy inspect number of value on categorical feature in DataFrame
class ValueCountFeatureInspection(DataInspectionStrategy):
    """
    Prints the number of value present on categorical column

    Parameters:
    df (pd.DataFrame): The dataframe to be inspected

    Returns:
    None: Prints the number of value in each categorical feature of dataframe
    """
    def inspect(self, df: pd.DataFrame):
        categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
        for feature in categorical_features:
            print(df[feature].value_counts())
            print("-------------------------------------------")


# Concrete Strategy class for Null Values Inspection
# ----------------------------------------------
# This strategy inspect to find number of null value in each column of dataframe
class IsNullFeatureInspection(DataInspectionStrategy):
    """
    Prints the number of null value present on each column

    Parameters:
    df (pd.DataFrame): The dataframe to be inspected

    Returns:
    None: Prints number of duplicate rows in the dataset.
    """
    def inspect(self, df: pd.DataFrame):
        print("\nNumber of Null Values in each feature:")
        print(df.isnull().sum())


# Concrete Strategy class for Duplicated Values Inspection
# ----------------------------------------------
# This strategy inspect to find number of duplicated value in each column of dataframe
class DuplicatedFeatureInspection(DataInspectionStrategy):
    """
    Prints the number of Duplicated value present on each column

    Parameters:
    df (pd.DataFrame): The dataframe to be inspected

    Returns:
    None: Prints the total number of Duplicated value in each feature of dataframe
    """
    def inspect(self, df: pd.DataFrame):
        print("\nSummary of Duplicated Values in each feature:")
        print(df.duplicated().sum())


# Concrete Strategy class for unique value inspection 
# ----------------------------------------------
# This strategy inspect number of unique value in each feature of dataframe
class NuUniqueValueInspection(DataInspectionStrategy):
    """
    Prints the number of unique value present on each column

    Parameters:
    df (pd.DataFrame): The dataframe to be inspected

    Returns:
    None: Prints the total number of unique value in each feature of dataframe
    """
    def inspect(self, df: pd.DataFrame):
        print("\nNumber of Unique Values in each colunm:")
        print(df.nunique())

# Concrete Strategy class for display unique value inspection 
# ----------------------------------------------
# This strategy inspect unique value in each feature of dataframe
class DisplayUniqueCatValueInspection(DataInspectionStrategy):
    """
    Prints the unique value present on each column

    Parameters:
    df (pd.DataFrame): The dataframe to be inspected

    Returns:
    None: Prints all unique value in each feature of dataframe
    """
    def inspect(self, df: pd.DataFrame):
        print("\nUnique Values in each colunm:")
        print(df.apply(lambda col: col.unique()))


class DiscreteNumericalFeature(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
        discrete_feature = [feature for feature in numerical_features if len(df[feature].unique()) <= 25]
        print(f"We have {len(discrete_feature)} discrete feature :", discrete_feature)
        return discrete_feature
    
class ContinuousNumericalFeature(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
        continuous_features = [feature for feature in numerical_features if len(df[feature].unique()) > 25]
        print(f"We have {len(continuous_features)} continuous feature :", continuous_features)
        return continuous_features


# Context Class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows you to switch between different data inspection strategies.
class DataInspector:
    def __init__(self):
        """
        Initilize the DataInspector strategy
        """
        self._strategy = None

    def execute_inspection(self, strategy_class, df: pd.DataFrame, **kwargs):
        """
        Sets and executes the specified inspection strategy.
        
        Parameters:
        strategy_class (class): The strategy class to instantiate and use.
        df (pd.DataFrame): The dataframe to be inspected.
        kwargs: Additional arguments for the strategy (e.g., n for number of rows).
        """

        # Print kwargs to see the additional arguments passed
        # print("Additional arguments (kwargs):", kwargs)
        self._strategy = strategy_class(**kwargs)
        return self._strategy.inspect(df)


# # Usage example
# if __name__ == "__main__":
#     # Sample DataFrame for demonstration
#     data = {'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e']}
#     df = pd.DataFrame(data)

#     # Initialize DataInspector
#     data_inspector = DataInspector()

#      # Call each strategy dynamically on the DataInspector instance
#     data_inspector.execute_inspection(HeadInspectionStrategy, df, n=3)  # Displays first 3 rows
#     data_inspector.execute_inspection(TailInspectionStrategy, df, n=2)  # Displays last 2 rows
#     data_inspector.execute_inspection(ShapeInspectionStrategy, df)      # Displays DataFrame shape