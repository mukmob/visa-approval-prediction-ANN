import os
from pathlib import Path

project_name = 'us_visa'
src = 'src'
notebook = 'notebook_research'

list_of_files = [
    f"config/model.yaml",
    f"config/schema.yaml",
    f"flowcharts/__init__.py",
    f"{notebook}/analysis_src/basic_data_inspection.py",
    f"{notebook}/analysis_src/univariate_analysis.py",
    f"{notebook}/analysis_src/bivariate_analysis.py",
    f"{notebook}/analysis_src/multivariate_analysis.py",
    f"{notebook}/analysis_src/missing_value_analysis.py",
    f"{notebook}/analysis_src/outliers_value_analysis.py",
    f"{notebook}/EDA.ipynb",
    f"static/css/style.css",
    f"templates/index.html",
    f"{project_name}/__init__.py",
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/cloud_storage/aws_storage.py",
    f"{project_name}/{src}/components/__init__.py",
    f"{project_name}/{src}/components/data_ingestion.py",
    f"{project_name}/{src}/components/data_validation.py",
    f"{project_name}/{src}/components/data_transformation.py", # Feature Engineering
    f"{project_name}/{src}/components/model_trainer.py",
    f"{project_name}/{src}/components/model_evaluation.py",
    f"{project_name}/{src}/components/model_pusher.py",
    f"{project_name}/{src}/components/handle_missing_values.py",
    f"{project_name}/{src}/components/outliers_detections.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/aws_connection.py",
    f"{project_name}/configuration/mongo_db_connection.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/{src}/data_access/__init__.py",
    f"{project_name}/{src}/data_access/ingest_data.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/estimator.py",
    f"{project_name}/entity/s3_estimator.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/{src}/pipelines/__init__.py",
    f"{project_name}/{src}/pipelines/prediction_pipeline.py",
    f"{project_name}/{src}/pipelines/training_pipeline.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py"
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
    else:
        print(f"file is already present at: {file_path}")


