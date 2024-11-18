import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import plotly.express as px

class MultivariateAnalysis:
    def __init__(self):
        pass
    
    # Shows pairwise scatter plots between multiple numerical features.
    def pair_plot(self, df, features, hue=None):
        """
        Displays pairwise scatter plots for multiple numerical features.
        """
        sns.pairplot(df, hue=hue, diag_kind='kde', corner=False)
        plt.suptitle("Pair Plot", y=1.02)
        plt.tight_layout()
        plt.show()

    # Displays correlation among multiple features.
    def heatmap_correlation(self, df, features):
        """
        Displays a heatmap of correlations among numerical features.
        """
        correlation = df[features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    #  Plots relationships among three numerical features.
    def scatter_3d(self, df, feature_x, feature_y, feature_z, hue=None):
        """
        Creates a 3D scatter plot for three numerical features and a categorical hue.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature_x (str): The feature for the x-axis.
        feature_y (str): The feature for the y-axis.
        feature_z (str): The feature for the z-axis.
        hue (str, optional): The categorical feature for color coding.

        Returns:
        None: Displays the 3D scatter plot.
        """

        fig = plt.figure(figsize=(8, 6), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        
        if hue:
            # Encode hue into numeric values
            hue_labels, hue_encoded = np.unique(df[hue], return_inverse=True)
            scatter = ax.scatter(
                df[feature_x],
                df[feature_y],
                df[feature_z],
                c=hue_encoded,
                cmap='viridis',
                s=50
            )
            # Generate legend handles and labels
            handles = scatter.legend_elements()[0]
            ax.legend(handles, labels=hue_labels, title=hue)
        else:
            scatter = ax.scatter(
                df[feature_x],
                df[feature_y],
                df[feature_z],
                c='b',
                s=50
            )
        
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_zlabel(feature_z)
        plt.title("3D Scatter Plot")
        plt.show()

    # Shows multivariate data by plotting each feature as a vertical line.
    def parallel_coordinates_plot(self, df, features, class_column):
        """
        Displays a parallel coordinates plot for multivariate data.
        """
        plt.figure(figsize=(12, 6))
        parallel_coordinates(df[features + [class_column]], class_column, color=sns.color_palette("tab10"))
        plt.title("Parallel Coordinates Plot")
        plt.tight_layout()
        plt.show()

    # A heatmap with hierarchical clustering to visualize similarity patterns.
    def cluster_map(self, df, features):
        """
        Displays a cluster map for numerical features with hierarchical clustering.
        """
        sns.clustermap(df[features].corr(), annot=True, cmap="coolwarm", figsize=(10, 8))
        plt.title("Cluster Map")
        plt.tight_layout()
        plt.show()

    # Displays multivariate data in a radial chart, useful for profile comparisons.
    def radar_plot(self, df, categories, row_label):
        """
        Displays a radar plot for multivariate data.
        """
        # Normalizing data for the radar plot
        data = df[categories].iloc[row_label].values.flatten()
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        data = np.concatenate((data, [data[0]]))  # Close the plot
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, data, color='blue', alpha=0.25)
        ax.plot(angles, data, color='blue', linewidth=2)
        ax.set_yticks([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title("Radar Plot")
        plt.tight_layout()
        plt.show()

    # Similar to a scatter plot but adds a third feature as bubble size.
    def bubble_plot(self, df, feature_x, feature_y, size, hue=None):
        """
        Displays a bubble plot where size of the bubbles indicates the value of a third feature.
        """
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x=feature_x, y=feature_y, size=size, hue=hue, sizes=(50, 500), alpha=0.6)
        plt.title(f"Bubble Plot: {feature_x} vs {feature_y}")
        plt.tight_layout()
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Example DataFrame
    df = sns.load_dataset('iris')
    print(df.head())

    # Initialize the class
    multivariate_analysis = MultivariateAnalysis()

    # # Pair Plot
    # multivariate_analysis.pair_plot(df, features=['sepal_length',  'sepal_width',  'petal_length',  'petal_width'], hue='species')

    # # Heatmap of Correlations
    # multivariate_analysis.heatmap_correlation(df, features=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    # # 3D Scatter Plot
    multivariate_analysis.scatter_3d(df, feature_x='sepal_length', feature_y='sepal_width', feature_z='petal_length', hue='species')

    # # Parallel Coordinates Plot
    # multivariate_analysis.parallel_coordinates_plot(df, features=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], class_column='species')

    # # Cluster Map
    # multivariate_analysis.cluster_map(df, features=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    # # Radar Plot
    # multivariate_analysis.radar_plot(df, categories=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], row_label=0)

    # # Bubble Plot
    # multivariate_analysis.bubble_plot(df, feature_x='sepal_length', feature_y='sepal_width', size='petal_length', hue='species')
