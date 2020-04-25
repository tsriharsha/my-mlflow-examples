# Databricks notebook source
def inject_mlrun_params(mlflow):
  import json
  if os.environ.get("GIT_REPO", None) != None:
    mlflow.set_tag("GIT_REPO", os.environ.get("GIT_REPO"))
  if os.environ.get("COMMIT_HASH", None) != None:
    mlflow.set_tag("COMMIT_HASH", os.environ.get("COMMIT_HASH"))
  if os.environ.get("GIT_REPO", None) != None:
    mlflow.set_tag("BRANCH", os.environ.get("BRANCH"))
  if os.environ.get("PARAMS_JSON_STRING", None) != None:
    params = json.loads(os.environ.get("PARAMS_JSON_STRING"))
    mlflow.log_params(params)
    
def get_mlrun_params():
  import os
  import json
  if os.environ.get("PARAMS_JSON_STRING", None) != None:
    return json.loads(os.environ.get("PARAMS_JSON_STRING", "{}"))
  else:
    return {}

# COMMAND ----------

1/0

# COMMAND ----------

def get_latest_version(delta_table_path):
  from delta.tables import DeltaTable  
  delta_table = DeltaTable.forPath(spark, data_path)
  return delta_table.history(1).select("version").collect()[0].version  

mlrun_params = get_mlrun_params()
data_path = mlrun_params.get("DELTA_LOAN_RISK_PATH", "dbfs:/tmp/loan_risk_labeled_delta_sri")
print("USING PATH: {}".format(data_path))
data_version = mlrun_params.get("DELTA_LOAN_RISK_VERSION", get_latest_version(data_path))
print("USING VERSION: {}".format(data_version))

# COMMAND ----------

# Use the latest version of the table by default, unless a version parameter is explicitly provided
loan_stats = spark.read.format("delta").option("versionAsOf", data_version).load(data_path)  

# Review data
display(loan_stats)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import mlflow


def _fit_crossvalidator(train, features, target, version):
  """
  Helper function that fits a CrossValidator model to predict a binary label
  `target` on the passed-in training DataFrame using the columns in `features`
  :param: train: Spark DataFrame containing training data
  :param: features: List of strings containing column names to use as features from `train`
  :param: target: String name of binary target column of `train` to predict
  """
  train = train.select(features + [target])
  model_matrix_stages = [
    Imputer(inputCols = features, outputCols = features),
    VectorAssembler(inputCols=features, outputCol="features"),
    StringIndexer(inputCol="bad_loan", outputCol="label")
  ]
  lr = LogisticRegression(maxIter=10, elasticNetParam=0.5, featuresCol = "features")
  pipeline = Pipeline(stages=model_matrix_stages + [lr])
  paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
  crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(),
                            numFolds=5)
  
  import matplotlib.pyplot as plt
  from mlflow import spark as mlflow_spark
  from mlflow import sklearn as mlflow_sk
  
  with mlflow.start_run():
    inject_mlrun_params(mlflow)
    cvModel = crossval.fit(train)
    best_model = cvModel.bestModel

    roc = best_model.stages[len(best_model.stages)-1].summary.roc.toPandas()
    fig1 = plt.figure()
    fig1.clf()
    plt.clf()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    fig1.savefig("roc.png")
    mlflow.log_artifact("roc.png")
    fig1.clf()
    plt.clf()

    lr_summary = best_model.stages[len(best_model.stages)-1].summary
    mlflow.log_metric("accuracy", lr_summary.accuracy)
    mlflow.log_metric("weightedFalsePositiveRate", lr_summary.weightedFalsePositiveRate)
    mlflow.log_metric("weightedFalsePositiveRate", lr_summary.weightedFalsePositiveRate)
    mlflow.log_metric("weightedFMeasure", lr_summary.weightedFMeasure())
    mlflow.log_metric("weightedPrecision", lr_summary.weightedPrecision)
    mlflow.log_metric("weightedRecall", lr_summary.weightedRecall)
    
    mlflow_spark.log_model(best_model, "loan-classifier-mllib")
    
    return best_model

# COMMAND ----------

# Fit model & display ROC
features = mlrun_params.get("FEATURES", ["loan_amnt",  "annual_inc", "dti", "delinq_2yrs","total_acc", "credit_length_in_years"])
glm_model = _fit_crossvalidator(loan_stats, features, target="bad_loan", version=data_version)
lr_summary = glm_model.stages[len(glm_model.stages)-1].summary
display(lr_summary.roc)

# COMMAND ----------

print("ML Pipeline accuracy: %s" % lr_summary.accuracy)