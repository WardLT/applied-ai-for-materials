# Homework: QSAR Models

The goal of this homework is to help establish the advantages and limitations of QSAR models with shallow machine learning methods.
You will build a few different QSAR models for the band gap energy of molecules to see the tradeoffs between different methods.

### Introduction

This homework is closely tied with the [chemoinformatics module](https://github.com/WardLT/applied-ai-for-materials/tree/main/molecular-property-prediction/chemoinformatics) from the course GitHub.
The computational environment and code snippets from the examples are going to be particularly useful for this assignment, by intention.
It is highly recommended to review them before completing the assignment and, especially, to use the provided computational environment (see: [Installation Instructions](https://github.com/WardLT/applied-ai-for-materials/tree/main/molecular-property-prediction#installation)).

Assignments will require you to provide text, plots and code to answer each question.
All 3 types of content can be stored in a Jupyter notebook, so we _highly_ recommend using that format for turning in your assignment.

### Chemical Descriptors

We are going to explore a few different machine learning models through predicting the band gap energy of molecules from their molecular strcutre.

First, load the *full* subset of [QM9](http://quantum-machine.org/datasets/#qm9) available [through our GitHub](https://github.com/WardLT/applied-ai-for-materials/blob/main/molecular-property-prediction/datasets/qm9.json.gz) and compute features using Mordred. You will need to convert all data to a float (see [`DataFrame.astype`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html)) and drop missing values, but do not use any other dimensionality reduction. Once complete, make a test set of 2000 molecules and:

1. Train a [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV) model using the features with and without reducing the dimensionality to 8 features using PCA. Then, train a [RandomForest model](https://scikit-learn.org/stable/modules/ensemble.html) using the default settings. Plot the change in mean absolute error (MAE) with respect to training set sizes: 10, 100, 1000, 10000. How do the results compare? Why do some models continue to improve at large training set sizes while others do not?


2. Now we look at two models specifically in the previous question, the LASSO without PCA and RandomForest trained on $10^4$ data points. Rank the top features for LASSO (using coefficients) and random forest (using the assigned feature scores) (Hint: [np.argsort](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)).

    1. Compare the top 10 features with lasso and random forest (Hint: [Python set logic](https://www.w3schools.com/python/python_sets_join.asp)). Are any the same? Why is this expected?
    1. Assess the correlation between top 10 features for lasso and random forest. What does this imply about how to interpret the most important features?
    1. Re-fit the model another time for one of the models, on a newly sampled set of $10^4$ data points. Do the results change between runs? 
    
   Describe what these results mean for interpreting the features of machine learning models. 

3. Discuss the relative advantages of RandomForest versus Linear Regression versus Linear Regression with PCA. 


### Molecular Fingerprints

We are going to do some experiments with improving Molecular Fingerprints.

1. Create a training set of 1000 entries. Train a k-Nearest Neighbors (kNN) regressor model using a Jaccard distance metric based on 128-length Morgan fingerprint with a radius of 3. Plot how the performance on the model (on a test set of 2000 entries) changes as you increase the number of neighbors used in kNN from 1 to $2^7$ by a factor of 2 each time. Explain why the MAE improves when increasing from 1 and then worsens as you increase past $2^4$.

2. Add a step in the model Pipeline from Step 1 that uses [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) with a Random Forest model to reduce the number of features to 32 in 4 steps. Compare the MAE versus number of neighbors to kNN without feature selection.

3. Why would the model with the feature selection perform better? In general terms, explain the disadvantage of using a general-purpose distnace metrics such as fingerprints and how must one must account for that.
