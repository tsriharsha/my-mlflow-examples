# Databricks notebook source
def print_metrics(alpha, l1_ratio, rmse, mae, r2):
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

# COMMAND ----------

def alpha_l1(in_alpha, in_l1_ratio):
  if float(in_alpha) is None:
    alpha = 0.05
  else:
    alpha = float(in_alpha)
    
  if float(in_l1_ratio) is None:
    l1_ratio = 0.05
  else:
    l1_ratio = float(in_l1_ratio)
  return alpha, l1_ratio

# COMMAND ----------

def split_x_y(data):
  from sklearn.model_selection import train_test_split
  
  # Split the data into training and test sets. (0.75, 0.25) split.
  train, test = train_test_split(data)
  train_x = train.drop(["progression"], axis=1)
  test_x = test.drop(["progression"], axis=1)
  train_y = train[["progression"]]
  test_y = test[["progression"]]
  return train_x, train_y, test_x, test_y

# COMMAND ----------

def ignore_warnings():  
  import warnings
  warnings.filterwarnings("ignore")

# COMMAND ----------

def diabetes():
  import pandas as pd
  import numpy as np
  from sklearn import datasets
  
  np.random.seed(40)
    
  # Load Diabetes datasets
  diabetes = datasets.load_diabetes()
  X = diabetes.data
  y = diabetes.target

  # Create pandas DataFrame for sklearn ElasticNet linear_model
  Y = np.array([y]).transpose()
  d = np.concatenate((X, Y), axis=1)
  cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
  data = pd.DataFrame(d, columns=cols)
  return data, X, y

def eval_metrics(actual, pred):
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import numpy as np
  rmse = np.sqrt(mean_squared_error(actual, pred))
  mae = mean_absolute_error(actual, pred)
  r2 = r2_score(actual, pred)
  return rmse, mae, r2

def compute_paths(X, y, eps, path):
  import os
  from sklearn.linear_model import lasso_path, enet_path
  from sklearn.linear_model import lasso_path, enet_path
  import matplotlib.pyplot as plt
  import numpy as np
  from itertools import cycle

  print("Computing regularization path using the lasso...")
  alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)
  print("Computing regularization path using the positive lasso...")
  alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(X, y, eps, positive=True, fit_intercept=False)
  print("Computing regularization path using the elastic net...")
  alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)
  print("Computing regularization path using the positive elastic net...")
  alphas_positive_enet, coefs_positive_enet, _ = enet_path(X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)
  
  # Lasso and Elastic-Net Paths
  fig1 = plt.figure(1)
  ax = plt.gca()

  colors = cycle(['b', 'r', 'g', 'c', 'k'])
  neg_log_alphas_lasso = -np.log10(alphas_lasso)
  neg_log_alphas_enet = -np.log10(alphas_enet)
  for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

  plt.xlabel('-Log(alpha)')
  plt.ylabel('coefficients')
  plt.title('Lasso and Elastic-Net Paths')
  plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
  plt.axis('tight')

  # Lasso and Positive Lasso
  fig2 = plt.figure(2)
  ax = plt.gca()
  neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
  for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)

  plt.xlabel('-Log(alpha)')
  plt.ylabel('coefficients')
  plt.title('Lasso and positive Lasso')
  plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
  plt.axis('tight')

  # Elastic Net and Positive Elastic Net
  fig3 = plt.figure(3)
  ax = plt.gca()
  neg_log_alphas_positive_enet = -np.log10(alphas_positive_enet)
  for (coef_e, coef_pe, c) in zip(coefs_enet, coefs_positive_enet, colors):
    l1 = plt.plot(neg_log_alphas_enet, coef_e, c=c)
    l2 = plt.plot(neg_log_alphas_positive_enet, coef_pe, linestyle='--', c=c)

  plt.xlabel('-Log(alpha)')
  plt.ylabel('coefficients')
  plt.title('Elastic-Net and positive Elastic-Net')
  plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'), loc='lower left')
  plt.axis('tight')

  # Save figures
  if os.path.isfile(path + "plot1.png"):
    os.system("rm "+ path + "plot1.png")
  if os.path.isfile(path + "plot2.png"):
    os.system("rm "+ path + "plot2.png")
  if os.path.isfile(path +"plot3.png"):
    os.system("rm "+  path + "plot3.png")
  fig1.savefig(path +"plot1.png")
  fig2.savefig(path + "plot2.png")
  fig3.savefig(path + "plot3.png")