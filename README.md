# Binary Classification Problem with Machine Learning

## Solving Approach:

### 1) Ultimate Goal of the Assignment:
    This assignment is about solving a binary classification problem, and I need to come up with a binary classifier that classifies given instances
    as class 1(Positive) and class 0 (Negative) based on the numerical features provided.
    
    
### 2) Getting to know the Dataset:
    Before selecting any machine learning algorithm for the given task it is better to know and explore the dataset provided. We should look 
    for the possible errors present inside datasets. After analysing the data I had following findings.
    
    I) Training set and Test set is given with training csv having 3910 record or instances and test csv having 691 records.
    
    II) There were no Null values present in any training or test set, so there was no need to deal with Null values.
    
    III) All the features present were of numerical types with non-zero values greater than 0.0 to pretty large numbers.
    
    IV) training_set.csv comes with a lable "Y" having two categories (Binary Value) of '0' and '1', but test_set.csv has only instances or records with not 
    labels provided for them
    
    V) From the observation of the training and test dataset, It is found that feature values having are large variation, some varies between 0 to 5,
    but some varying between 0 to 1000, while few from 0 to 10000, and so on.
    
    VI) Most importantly, the dataset is imbalaned. It has 1534 instances belonging to class '1' and 2376 instances for class '0' having imbalance
    ration as 1.5489.
    
### 3) What Preprocessing techniques? and Why?
    I) I used Simple Histograms which helped to find the distribution of each features, density of them and in what proportions there are varying.
    II) KDE plot is vey important, it depicts the probability density at different values in a continuous variable.
    III) Box-Whiskers Plot, this plot are very important and gives interesting insights on dataset, it gives, 1st IQR(25th Percentile), 2nd IQR
    (median), 3rd IQR (75th Percentile), Upper bound, Lower bound, and Specially Outliers!!
    IV) From box plots, it is observed that the dataset has lot of outliers also few of them havinf very large values, hence giving scope for data 
    scaling or standardization.
    V) Manually, I found the number of features having values greater than 1.0. Some features are very much concentrated between 0 to 1.0 but few are 
    totally outside this range.
    
### 4) Feature Engineering and Feature Selection:
    I) In feature engineering, we can combine existing features or use domain knowledge to design completely new features. Here I haven't explored on engineering
    part, but focused on selection (though I removed only 1 of them!!)
    II) There are 57 numerical features, so I decided to remove highly correlated features, as highly correlated features causes redundancy in dataset.
    So it is always advisable to remove highly correlated features.
    III) I used Corr() function to find correlations between features with respect to another. And displayed them in the form of Correlational Matrix.
    IV) Due to large features, the matrix was pretty much messier!!. So I manually filter the features along with its highly correlated features list.
    I used 85% correlation threshold limit. 
    V) Only X32  and X34 were filtered out in this criterion, and decided to drop X32 (Just random decision, not based on P-Value).
    
### 5) Algorithm Selection and Tuning:
    I) Model selection has no strict rules, but decision is taken from considering number of factors, such as number of features vs number of instances,
    Linearity of data, speed, accuracy and so on.
    II) From the feature pairplots, we found that dataset is highly distributed and very few are linearly separable, so I decided to go with Non-Linear
    model like KNN, Decision Tree - Random Forest, XGBoost, SVM, etc,.
    III) Since total number of records are 3910 and features 57, so records >> features, here KNN, Kernel-SVM, Desision tree, Random Firest are good choice.
    IV) We have outliers in our data, so KNN and tree-based models are very robust to outliers.
    V) The given dataset is small, so I ignored training time criterion to filter models.
    VI) Finally I moved forward with KNN, Random Forest Classifer and XGBClassifier models.
    
### 6) Which accuravy measure to use? and Why?
    I) We are dealing with Binary Classification task, So I decided to include multiple measure to assess the quality of predictions and 
    performance of the models.
    II) Accuracy measures followed --> Model accuracy Score, Confusion Matrix, Precision Score, Recall Score, F1-Score, ROC_AUC Score, ROC Curve
    III) Accuracy Score - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations.
    IV) Confusion Matrix - Confusion matrix is a very popular measure used while solving classification problems. It can be applied to binary classification as well as for multiclass classification problems.
    Confusion matrices represent counts from predicted and actual values. It gives four numbers TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative).
    
              ---------------------------------------------------------------------------------------------------------------------------
              | True Negative | True Negative which shows the number of negative examples classified accurately | class '0' to class '0' |
              ---------------------------------------------------------------------------------------------------------------------------
              | True Positive |  True Positive which indicates the number of positive examples classified accurately| class '1' to class '1'
              ---------------------------------------------------------------------------------------------------------------------------------------------
              | False Positive | False Positive which shows the number of actual negative examples classified as positive | actual class '0' to class '1' |
              ---------------------------------------------------------------------------------------------------------------------------------------------
              | False Negative | False Negative value which shows the number of actual positive examples classified as negative | actual class '1' to class '0' |
              ---------------------------------------------------------------------------------------------------------------------------------------------------
    V) Precision Score - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 
                ----------------------------------------------------------------------
                | Precision = TP/TP+FP | Where, TP = True Positive, FP = False Positive
                ----------------------------------------------------------------------
    VI) Recall Score - This is also called 'Sensitivity'. It is the ratio of correctly predicted positive observations to the all observations in actual class.
                ----------------------------------------------------------------------
                | Recall = TP/TP+FN | Where, TP = True Positive, FN = False Negative |
                ----------------------------------------------------------------------
    VII) F1 Score - F1 Score is the weighted average of Precision and Recall. 
                ------------------------------------------------------------
                | F1 Score = 2*(Recall * Precision) / (Recall + Precision) |
                ------------------------------------------------------------
    VIII) ROC Curve - It is a chart that visualizes the tradeoff between true positive rate (TPR) and false positive rate (FPR). Basically, for every threshold, 
    we calculate TPR and FPR and plot it on one chart. The higher TPR and the lower FPR is for each threshold the better and so classifiers that have curves that 
    are more top-left-side are better.
    IX) ROC_AUC Score - ROC score is nothing but the area under ROC curve. The more it close to zero, better is our classifier algorithm.
    
### 7) How we can Improve further?
        -----------------------------------------------------------------------------------------------------------------------
        | Data Imbalance | we should reduce data imbalance issue so that model is not biased against any class |
        -----------------------------------------------------------------------------------------------------------------------------------
        | Remove Outliers | We can use box-whiskers plots, Z-score, IQR based filtering, Percentile, Winsorization, etc to remove outliers |
        ------------------------------------------------------------------------------------------------------------------------------------
        | Feature Engineering | We can combine several features with each other to create new features, Use Domain Knowledge |
        -----------------------------------------------------------------------------------------------------------------------
        | Reduce Dimensionality - Feature selection | We can use Principle Component Analysis (PCA), t-SNE to filter out most useful features having large variance |
        -------------------------------------------------------------------------------------------------------------------------------------------------------------
        | Hyper Parameter Tuning | We can play around different algorithms and hyper tune them with most optimum algorithm parameters to avoid overfitting |
        --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        | Deep Neural Networks | If we have huge dataset, neural networks are very effective to capture hidden representations from dataset with reduced interpretability of the model |
        --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------.
        
        
Please revert for any doubts. Thank You!!
