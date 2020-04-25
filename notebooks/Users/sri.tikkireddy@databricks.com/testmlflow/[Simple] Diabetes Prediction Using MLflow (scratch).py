# Databricks notebook source
# MAGIC %md ## MLflow Diabetes Quick Start
# MAGIC 
# MAGIC ![Imgur](https://i.imgur.com/moAWKNl.png)
# MAGIC 
# MAGIC We want to try training an ElasticNet linear regression model where we modify the `alpha` and `l1_ratio` parameters.
# MAGIC 
# MAGIC This notebook predicts the progression (quantitative measure of disease progression one year after baseline) based on BMI, blood pressure, etc. based on the *diabetes* dataset included in sklearn.

# COMMAND ----------

# MAGIC %run ./lib

# COMMAND ----------

# DBTITLE 1,Define SKL Python Function
import os
def train_diabetes(in_alpha, in_l1_ratio):
  from sklearn.linear_model import ElasticNet
  import mlflow
  import mlflow.sklearn
  ignore_warnings()

  alpha, l1_ratio = alpha_l1(in_alpha, in_l1_ratio)
  
  # Load diabetes dataset
  data, X, y = diabetes()

  # Split the data into training and test sets. (0.75, 0.25) split.
  train_x, train_y, test_x, test_y = split_x_y(data)
      
  with mlflow.start_run(experiment_id=os.environ.get("EXPERIMENT_ID")):
    mlflow.set_tag("GIT_REPO", os.environ.get("GIT_REPO", "notebook_run"))
    mlflow.set_tag("COMMIT_HASH", os.environ.get("COMMIT_HASH", "notebook_run"))
    mlflow.set_tag("BRANCH", os.environ.get("BRANCH", "notebook_run"))
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out ElasticNet model metrics
    print_metrics(alpha, l1_ratio, rmse, mae, r2)

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "model")

    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path
    path = "/dbfs/plots/"
    compute_paths(X, y, eps, path)

    # Log artifacts (output files)
    mlflow.log_artifacts(path)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.05, 0.05
train_diabetes(0.05, 0.05)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.02, 0.05
train_diabetes(0.02, 0.05)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.02, 0.1
train_diabetes(0.02, 0.1)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 0.1
train_diabetes(0.0001, 0.1)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.005, 0.1
train_diabetes(0.005, 0.2)

# COMMAND ----------

# MAGIC %md ## Let's look at the MLflow UI...

# COMMAND ----------

# ML Projects