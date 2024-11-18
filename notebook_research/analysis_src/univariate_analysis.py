from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from us_visa.exception import CustomException
from us_visa.logger import logging
import sys

# Abstract Base Class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies.
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str, plot_type: str):
        """
        Perform univariate analysis on a specific feature of the dataframe using a specified plot type.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.
        plot_type: The name of the plot.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass

# Concrete Strategy for Numerical Features
# -----------------------------------------
# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, plot_type: str):
        """
        Plots the specified type of plot for a numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a specific plot.
        """
        plt.figure(figsize=(6, 4), dpi = 200)

        plot_functions = {
            'histogram': self.histogram,
            'boxplot': self.boxplot,
            'density': self.density,
            'violin': self.violin,
            'kde': self.kde
        }

        # Store the value of plot type
        plot_func = plot_functions.get(plot_type)
        if plot_func:
            """
            The purpose of this line is to decouple the logic for selecting the appropriate
            plotting function from the actual plotting logic.
            """
            plot_func(df, feature)
            # This becomes:
            # self.histogram(df, feature) based on plot type
        else:
            raise CustomException(f"Unsupported plot type '{plot_type}' for numerical analysis.", sys)
    
    # Shows the distribution of the data.  
    def histogram(self, df, feature):
        sns.histplot(df[feature], kde=True, bins=50)
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # Identifies outliers and the spread of the data.
    def boxplot(self, df, feature):
        sns.boxplot(x=df[feature])
        plt.title(f"Box Plot of {feature}")
        plt.xlabel(feature)
        plt.tight_layout()
        plt.show()

    # A smooth curve that shows data distribution.
    def density(self, df, feature):
        sns.kdeplot(df[feature], fill=True)
        plt.title(f"Density Plot of {feature}")
        plt.xlabel(feature)
        plt.tight_layout()
        plt.show()

    # Combines box plot and density plot features.
    def violin(self, df, feature):
        sns.violinplot(x=df[feature])
        plt.title(f"Violin Plot of {feature}")
        plt.xlabel(feature)
        plt.tight_layout()
        plt.show()

    # KDE (Kernel Density Estimate) Plot - Visualizes the probability density function.
    def kde(self, df, feature):
        sns.kdeplot(df[feature], shade=True)
        plt.title(f"KDE Plot of {feature}")
        plt.xlabel(feature)
        plt.tight_layout()
        plt.show()

# Concrete Strategy for Categorical Features
# -------------------------------------------
# This strategy analyzes categorical features by plotting their frequency distribution.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, plot_type: str):
        """
        Plots the specified type of plot for a categorical feature..

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.
        plot_type: The name of specific plot type selected by user

        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        plt.figure(figsize=(6, 4), dpi = 200)

        plot_functions = {
            'bar': self.bar,
            'pie': self.pie,
            'count': self.count,
            'frequency_table': self.frequency_table
        }
        
        plot_func = plot_functions.get(plot_type)
        if plot_func:
            """
            The purpose of this line is to decouple the logic for selecting the appropriate
            plotting function from the actual plotting logic.
            """
            plot_func(df, feature)
        else:
            raise CustomException(f"Unsupported plot type '{plot_type}' for categorical analysis.", sys)
        
    # Shows the frequency of categories.
    def bar(self, df, feature):
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Bar Plot of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Represents proportions of categories.
    def pie(self, df, feature):
        df[feature].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
        plt.title(f"Pie Chart of {feature}")
        plt.ylabel("")  # Hide the y-label for pie chart
        plt.tight_layout()
        plt.show()

    # Counts of each category.
    def count(self, df, feature):
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Count Plot of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        # plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Displays counts of each category in tabular form.
    def frequency_table(self, df, feature):
        frequency_table = df[feature].value_counts()
        print(f"\nFrequency Table for {feature}:\n{frequency_table}")
        # No plot to show, only the frequency table is printed

# Concrete Strategy for Multiple Feature  Univariate Analysis Strategy
# ----------------------------------------------
# This class allows to display multiple feature into a single frame i.e. subplots.
class MultiPlotUnivariantAnalysis(UnivariateAnalysisStrategy):
    def __init__(self, features: list, n_cols:int = 2, plot_type = 'histogram'):
        """
        Initializes with a list of features and the number of columns for subplots.

        Parameters:
        features (list): List of feature names to analyze.
        n_cols (int): Number of columns for the subplot layout.
        plot_type (str): Type of plot to use for each feature.
        """
        self.features = features
        self.n_cols = n_cols
        self.plot_type = plot_type

    def analyze(self, df: pd.DataFrame, feature: str = None, plot_type: str = None):
        """
        Plots multiple features in a grid layout.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        """
        n_rows = (len(self.features) + self.n_cols - 1) // self.n_cols  # Calculate required rows
        fig, axes = plt.subplots(n_rows, self.n_cols, figsize=(self.n_cols * 6, n_rows * 5), dpi=200)
        axes = axes.flatten()

        # Dictionary to handle different plot types
        plot_functions = {
            'histogram': self.histogram,
            'boxplot': self.boxplot,
            'density': self.density,
            'violin': self.violin,
            'kde': self.kde,
            'bar': self.bar,
            'pie': self.pie,
            'count': self.count
        }

        for i, feature in enumerate(self.features):
            ax = axes[i]
            # The := operator is a concise way to assign and check a variable at the same time.
            if plot_func := plot_functions.get(self.plot_type):
                plot_func(df, feature, ax)  # Pass the axis to each plotting function
                ax.set_title(f"{self.plot_type.title()} of {feature}")
                ax.set_xlabel(feature)
                ax.set_ylabel("Density" if pd.api.types.is_numeric_dtype(df[feature]) else "Frequency")
 
            else:
                logging.warning(f"Unsupported plot type '{self.plot_type}' for feature '{feature}'.")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.show()


    def histogram(self, df, feature, ax):
        sns.histplot(df[feature], kde=True, bins=50, ax=ax)

    def boxplot(self, df, feature, ax):
        sns.boxplot(x=df[feature], ax=ax)

    def density(self, df, feature, ax):
        sns.kdeplot(df[feature], fill=True, ax=ax)

    def violin(self, df, feature, ax):
        sns.violinplot(x=df[feature], ax=ax)

    def kde(self, df, feature, ax):
        sns.kdeplot(df[feature], shade=True, ax=ax)

    def bar(self, df, feature, ax):
        p = sns.countplot(x=feature, data=df, ax=ax, palette="muted")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Add labels to each bar
        for patch in p.patches:
            height = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom')

    def pie(self, df, feature, ax):
        df[feature].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, ax=ax)
        ax.set_ylabel("")

    def count(self, df, feature, ax):
        p = sns.countplot(x=feature, data=df, ax=ax, palette="muted")
        # Add labels to each bar
        for patch in p.patches:
            height = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom')


# Context Class that uses a UnivariateAnalysisStrategy
# ----------------------------------------------------
# This class allows you to switch between different univariate analysis strategies.
class UnivariateAnalyzer:
    def __init__(self):
        """
        Initializes the UnivariateAnalyzer with no specific strategy set initially.

        Parameters:
        strategy: The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = None

    def execute_analysis(self, df: pd.DataFrame, feature: str = None, features: list = None, n_cols: int = 2, plot_type: str = 'histogram'):
        """
        Executes univariate analysis with specified plot type based on the feature's data type.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str, optional): A single feature for analysis.
        features (list, optional): A list of features for multiple analysis.
        n_cols (int, optional): Number of columns for layout when analyzing multiple features.
        plot_type (str): The type of plot to display.
        """
        if features:
            # Use MultipleUnivariateAnalysis for multiple features in a grid layout
            self._strategy = MultiPlotUnivariantAnalysis(features, n_cols, plot_type)
            self._strategy.analyze(df)
        elif feature:
            # Automatically select the strategy based on the feature's data type
            if pd.api.types.is_numeric_dtype(df[feature]):
                self._strategy = NumericalUnivariateAnalysis()
                self._strategy.analyze(df, feature, plot_type)
            else:
                self._strategy = CategoricalUnivariateAnalysis()
                self._strategy.analyze(df, feature, plot_type)
        
# Example usage   
# if __name__ == "__main__":
#     df = pd.read_csv("/Users/aadarsh/Desktop/Data Scientist/Projects/US-Visa-Approval-Prediction/us_visa/data/extracted_data/EasyVisa.csv")


#     # Initialize UnivariateAnalyzer
#     univariant_analyzer = UnivariateAnalyzer()

#     # Analyze a single numerical feature with different plot types
#     univariant_analyzer.execute_analysis(df, feature='no_of_employees', plot_type='histogram')
#     univariant_analyzer.execute_analysis(df, feature='no_of_employees', plot_type='boxplot')
#     univariant_analyzer.execute_analysis(df, feature='yr_of_estab', plot_type='density')

#     # Analyze a single categorical feature with different plot types
#     univariant_analyzer.execute_analysis(df, feature='case_status', plot_type='bar')
#     univariant_analyzer.execute_analysis(df, feature='full_time_position', plot_type='pie')

#     # Analyze multiple features at once with specified layout
#     univariant_analyzer.execute_analysis(df, features=['no_of_employees', 'case_status', 'unit_of_wage'], n_cols=2, plot_type='histogram')













