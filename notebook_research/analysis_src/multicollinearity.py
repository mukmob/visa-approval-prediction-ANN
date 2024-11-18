from abc import ABC, abstractmethod
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from us_visa.exception import CustomException
from us_visa.logger import logging
import sys
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

from us_visa.utils.main_utils import separate_numerical_and_categorical


# Abstract Base Class for MultiCollinearity Inspections Strategy
# -----------------------------------------------------
class MultiCollinearityInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame, correlation_threshold: float = None, vif_threshold: float = None, target_column: str = None):
        """
        Inspect for numerical collinearity using Correlation Matrix and VIF.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the numerical features to inspect.
        correlation_threshold (float): The threshold for filtering correlated features.
        vif_threshold (float): The threshold for filtering features with high VIF values.
        return: correlation and vif data
        """
        pass


class NumericalMultiCollinearityInspection(MultiCollinearityInspectionStrategy):
    def inspect(self, df: pd.DataFrame, correlation_threshold: float = None, vif_threshold: float = None, target_column: str = None):
        logging.info("Inspecting numerical collinearity:")
        try:
            numerical_df, _ = separate_numerical_and_categorical(df)
            df = numerical_df
        except:
            raise CustomException("Given dataframe features should be numeric", sys)

        # Calculate and display the correlation matrix
        corr_data = self.plot_correlation_matrix(df, correlation_threshold)
        
        # Calculate and display the VIF
        vif_data = self.calculate_vif(df, vif_threshold)
        return corr_data, vif_data
    
    def plot_correlation_matrix(self, df: pd.DataFrame, threshold: float):
        """
        Plot the correlation matrix and return a filtered list of correlated features based on the threshold.
        """
        correlation_matrix = df.corr()

        """
        This get_correlated_feature function will be use when target feature is numerical
        """
        # correlated_features = self.get_correlated_feature(correlation_matrix, threshold)
        
        # Plot the correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()
        
        # return correlation_matrix
        # return correlated_features
    
    def get_correlated_feature(self, corr_data, threshold):
        """
        Filter and return features that have a correlation value greater than the threshold.
        
        Parameters:
        corr_data (pd.DataFrame): The correlation data will be target feature.
        threshold (float): The correlation threshold to filter features.
        
        Returns:
        pd.DataFrame: A DataFrame containing the features and their correlation values above the threshold.
        """
        feature = []
        value = []
        
        for i , index in enumerate(corr_data.index):
            if abs(corr_data[index])>threshold:
                feature.append(index)
                value.append(corr_data[index])

        # Create and return a DataFrame with the filtered correlated features
        df = pd.DataFrame(data=value, index=feature, columns=['corr value'])
        return df
    
    def calculate_vif(self, df: pd.DataFrame, threshold: float):
        """
        Calculate and return the Variance Inflation Factor (VIF) and filter by the threshold.
        """
        # Add constant to the dataframe
        df_with_const = sm.add_constant(df)
        
        # Ensure only numeric columns are included
        df_with_const = df_with_const.select_dtypes(include=['float64', 'int64'])
        
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])]
        
        # Remove the constant term from the results
        vif_data = vif_data[vif_data["Feature"] != 'const']
        
        # Filter out features with high VIF (greater than threshold)
        vif_data = vif_data[vif_data["VIF"] > threshold]

        print("Variance Inflation Factor (VIF):")
        print(vif_data)
        return vif_data

# Concrete Strategy for Categorical Collinearity
class CategoricalMultiCollinearityInspection(MultiCollinearityInspectionStrategy):
    
    def inspect(self, df: pd.DataFrame, correlation_threshold: float = None, vif_threshold: float = None, target_column: str = None):
        logging.info("Inspecting categorical collinearity:")
        try:
            _, categorical_df = separate_numerical_and_categorical(df)
            self.check_categorical_dependence(categorical_df, target_column)
        except:
            raise CustomException("Given dataframe features should be categorical", sys)

    def check_categorical_dependence(self, df: pd.DataFrame, target_column):
        """
        Check for dependence between categorical features using Chi-square test.

        - A chi-squared test (also chi-square or χ2 test) is a statistical hypothesis test 
        that is valid to perform when the test statistic is chi-squared distributed under the 
        null hypothesis, specifically Pearson's chi-squared test.

        - A chi-square statistic is one way to show a relationship between two categorical variables.
        - Here we test correlation of Categorical columns with Target column.

        - Null Hypothesis (Ho): The Feature is independent of target column (No-Correlation)
        - Alternative Hypothesis (H1): The Feature and Target column are not independent(dependent) (Correalted)
        """
        categorical_features = df.select_dtypes(include=['object', 'category']).columns
        chi2_test = []
        for feature in categorical_features:
            if chi2_contingency(pd.crosstab(df[target_column], df[feature]))[1] < 0.05:
                chi2_test.append('Reject Null Hypothesis')
            else:
                chi2_test.append('Fail to Reject Null Hypothesis')
        result = pd.DataFrame(data=[categorical_features, chi2_test]).T
        result.columns = ['Column', 'Hypothesis Result']
        print(result)

        # Check for dependence between categorical features using Chi-square test
        dependence_result = []
        for col1 in categorical_features:
            for col2 in categorical_features:
                if col1 != col2:
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    # Collect the result in a Dictionary
                    dependence_result.append({
                        'Feature 1': col1,
                        'Feature 2': col2,
                        'Chi-Square': chi2,
                        'P-Value': p,
                        'Degrees of Freedom': dof,
                        'Dependency': 'Dependent' if p < 0.05 else 'Independent'
                    })
                    if p < 0.05:  # If p-value is less than 0.05, features are dependent
                        print(f"Features '{col1}' and '{col2}' are dependent (Chi-square p-value: {p:.3f})")
                    else:
                        print(f"Features '{col1}' and '{col2}' are independent (Chi-square p-value: {p:.3f})")

        # Convert the results to a DataFrame
        dependence_result_df = pd.DataFrame(dependence_result)
        # print(dependence_result_df)

# Context Class to Use the Strategy
class  MultiCollinearityInspector:
    def __init__(self, strategy: MultiCollinearityInspectionStrategy = None):
        # Initialize with a default strategy or None
        self._strategy = strategy   

    def set_strategy(self, strategy: MultiCollinearityInspectionStrategy):
        """
        Set or change the strategy dynamically.
        
        Parameters:
        strategy (MultiCollinearityInspectionStrategy): The strategy to set.
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame, correlation_threshold: float = None, vif_threshold: float = None, target_column: str = None):
        if self._strategy is None:
            raise CustomException("Strategy is not set. Use `set_strategy` to set a strategy before executing inspection.", sys)
        return self._strategy.inspect(df, correlation_threshold, vif_threshold, target_column)

"""
# Example usage
if __name__ == "__main__":
    df = pd.read_csv("/Users/aadarsh/Desktop/Data Scientist/Projects/US-Visa-Approval-Prediction/us_visa/data/extracted_data/EasyVisa.csv")
    
    correlation_threshold = 0.8
    vif_threshold = 5.0
    target_column = 'case_status'

     # Initialize the context class without a strategy
    inspector = MultiCollinearityInspector()

    # inspector.set_strategy(NumericalMultiCollinearityInspection())
    # inspector.execute_inspection(df, correlation_threshold, vif_threshold, target_column)
    
    inspector.set_strategy(CategoricalMultiCollinearityInspection())
    inspector.execute_inspection(df, correlation_threshold, vif_threshold, target_column)

"""
    