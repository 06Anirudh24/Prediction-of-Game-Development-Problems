# Prediction of Video Game Development Problems

A video game’s post-mortem describes its history, team goals and discusses what went right and wrong with the development. Despite its significance, there is little understanding about the challenges faced by programmers because post-mortems are informally written and not properly maintained, leading to a luck of trustworthiness. In this study, a systematic analysis has been performed on different problems faced in game development. The need for automation and machine learning arises as it could aid developers easily identify the exact problem from the description, and hence be able to easily find a solution. This work could also help developers in identifying frequent mistakes that could be avoided, and act as a starting point to further consider game development in the context of software engineering. For the purpose of this work, game development problems have been divided into three groups - production, business, management.

The objective of this work is to build a classifier that predicts the group of video game development problem when given a problem quote/description. For this purpose, various word embedding techniques, feature selection techniques, and machine learning and ensemble learning classifiers are built. A comparative study is performed on the various models using boxplots, AUC scores, and Friedman's non-parametric test. 

The work makes use of the following models: -
Word Embeddings - TFIDF, Skipgram, CBOW, Word2Vec, BERT, GloVe, and FastText.
SMOTE - Synthetic Minority Oversampling Technique (Used to account for class imbalance in the dataset). 
Feature Section Techniques: - Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA) and Analysis of Variance (ANOVA).
Machine Learning Classifiers - K-Nearest Neighbours (KNN), Support Vector Classifier (SVC), Naive Bayes Classifier (NBC), Decision Tree (DT), and Random Forest (RF) .
Ensemble Learning Classifiers - Bagging, Extra Trees, AdaBoost, GradBoost, and Major Voting Ensemble. 

The empirical results of this work has been published at the ICON-2021 Conference. 
