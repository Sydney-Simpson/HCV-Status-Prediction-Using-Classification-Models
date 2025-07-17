# HCV-Status-Prediction-Using-Classification-Models

# Table of Contents
[Background](#Background) </br>
[Exploratory Analysis](#Exploratory-Analysis) </br>
[Data Cleaning and Manipulation](#Data-Cleaning-and-Manipulation) </br>
[Decision Tree](#Decision-Tree) </br>
[Random Forest](#Random-Forest) </br>
[Random Forest Model Features](#Random-Forest-Model-Features) </br>
[Gradient Boosting](#Gradient-Boosting)

# Background
Hepatitis C Virus (HCV) is a bloodborne pathogen and a causitive agent of liver disease. It causes a chronic progressive infection, leading to fibrosis and eventually cirrhosis of the liver. Until very recently HCV was incurable. However, it remains a public health concern.

### Dataset
This dataset comes from the [UC Irvine Machine Learning Repositiory](https://archive.ics.uci.edu/dataset/571/hcv+data). It contains laboratory values for blood donors and Hepatitis C patients, as well as some demographic information. There are 615 instances, and 12 features.

### Goal
The goal of this project is to develop a model to predict whether a sample comes from a patient with Hepatitis C or from a healthy donor. The model was developed using Random Forest Classification, which is a machine learning technique that works well for logistic problems such as whether a sample is from a person with or without HCV infection. However, random forest classifications are not limited to problems with binary (in this case yes or no) answers.

# Exploratory Analysis
The data is read in in two separate files. One file contains all features, the other contains targets ( Categories based on disease state). Feature information and descriptive statistics can be seen below.



``` python

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 615 entries, 0 to 614
Data columns (total 12 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Age     615 non-null    int64  
 1   Sex     615 non-null    object 
 2   ALB     614 non-null    float64
 3   ALP     597 non-null    float64
 4   AST     615 non-null    float64
 5   BIL     615 non-null    float64
 6   CHE     615 non-null    float64
 7   CHOL    605 non-null    float64
 8   CREA    615 non-null    float64
 9   CGT     615 non-null    float64
 10  PROT    614 non-null    float64
 11  ALT     614 non-null    float64
dtypes: float64(10), int64(1), object(1)
memory usage: 57.8+ KB\

```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>ALB</th>
      <th>ALP</th>
      <th>AST</th>
      <th>BIL</th>
      <th>CHE</th>
      <th>CHOL</th>
      <th>CREA</th>
      <th>CGT</th>
      <th>PROT</th>
      <th>ALT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>615.000000</td>
      <td>614.000000</td>
      <td>597.000000</td>
      <td>615.000000</td>
      <td>615.000000</td>
      <td>615.000000</td>
      <td>605.000000</td>
      <td>615.000000</td>
      <td>615.000000</td>
      <td>614.000000</td>
      <td>614.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>47.408130</td>
      <td>41.620195</td>
      <td>68.283920</td>
      <td>34.786341</td>
      <td>11.396748</td>
      <td>8.196634</td>
      <td>5.368099</td>
      <td>81.287805</td>
      <td>39.533171</td>
      <td>72.044137</td>
      <td>28.450814</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.055105</td>
      <td>5.780629</td>
      <td>26.028315</td>
      <td>33.090690</td>
      <td>19.673150</td>
      <td>2.205657</td>
      <td>1.132728</td>
      <td>49.756166</td>
      <td>54.661071</td>
      <td>5.402636</td>
      <td>25.469689</td>
    </tr>
    <tr>
      <th>min</th>
      <td>19.000000</td>
      <td>14.900000</td>
      <td>11.300000</td>
      <td>10.600000</td>
      <td>0.800000</td>
      <td>1.420000</td>
      <td>1.430000</td>
      <td>8.000000</td>
      <td>4.500000</td>
      <td>44.800000</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.000000</td>
      <td>38.800000</td>
      <td>52.500000</td>
      <td>21.600000</td>
      <td>5.300000</td>
      <td>6.935000</td>
      <td>4.610000</td>
      <td>67.000000</td>
      <td>15.700000</td>
      <td>69.300000</td>
      <td>16.400000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>47.000000</td>
      <td>41.950000</td>
      <td>66.200000</td>
      <td>25.900000</td>
      <td>7.300000</td>
      <td>8.260000</td>
      <td>5.300000</td>
      <td>77.000000</td>
      <td>23.300000</td>
      <td>72.200000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>54.000000</td>
      <td>45.200000</td>
      <td>80.100000</td>
      <td>32.900000</td>
      <td>11.200000</td>
      <td>9.590000</td>
      <td>6.060000</td>
      <td>88.000000</td>
      <td>40.200000</td>
      <td>75.400000</td>
      <td>33.075000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77.000000</td>
      <td>82.200000</td>
      <td>416.600000</td>
      <td>324.000000</td>
      <td>254.000000</td>
      <td>16.410000</td>
      <td>9.670000</td>
      <td>1079.100000</td>
      <td>650.900000</td>
      <td>90.000000</td>
      <td>325.300000</td>
    </tr>
  </tbody>
</table>
</div>

Meanwhile the target file contains a single column of data titled 'Category'.

```python

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 615 entries, 0 to 614
Data columns (total 1 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Category  615 non-null    object
dtypes: object(1)
memory usage: 4.9+ KB

```
Data are placed under the following 5 classifications based on disease state.
* 0=Blood Donor - Samples obtained from blood donors, not HCV patients
* 0s=suspect Blood Donor - Samples obtained from blood donors, not HCV patients- however they may have HCV. Unfortunately there is not elboration on why this is suspected
* 1=Hepatitis - Samples from HCV positive patients without Fibrosis or Cirrhosis
* 2=Fibrosis - Samples from HCV positive patients with Fibrosis
* 3=Cirrhosis - Samples from HCV positive patients with Cirrhosis

## Distribution of data by age
First, we combine the two dataframes into a single dataframe, this makes it easier to quickly look through the data. 

First we use a histogram to see how our disease state is distributed by age. 
<img width="842" height="545" alt="output_11_1" src="https://github.com/user-attachments/assets/b124028c-9e0d-4b0a-95f3-7681391e9d2d" />

There are a few insights from this distribution:
1) There are no Blood donor (HCV-) samples below age 32. Logically, this is due to sampling. Just because there are samples from infected person below the age of 32 doesn't mean that all samples below this age will be infected. We'll need to consider this if age is determined to be important in the final model.
2) HCV positive individuals without Fibrosis or Cirrhosis are likely to be younger. This makes sense logically as liver damage takes time to accumulate.

Next, we look at how instances there are for each disease state, as well as how disease states are split among our single demographic variable- sex.

<img width="1005" height="622" alt="output_15_1" src="https://github.com/user-attachments/assets/4ee45462-af50-421d-851e-46a2d0d16b39" />

1) There are very few suspect Blood Donor samples, and only one is female.
2) There are far fewer HCV+ samples than HCV- samples. This may cause issues with sample imbalance. Thankfully Random Forest Classifications are not known to be prone to issues with sample imbalance. However- as we will be separating our data into a training set and a test set, we do need to ensure that there are enough samples for each disease state to both train and test our model.

# Data Cleaning and Manipulation
Due to the issues noted above in the exploratory analysis we make the following changes to the data:
* We remove the suspect blood donor samples from the analysis. We do this for two reasons
  1) They are negative but suspected of being positive. We don't have information on why they are suspected of being positive. However, <b> since their status is somewhat in limbo there is no 'correct' classification. That makes them impossible to use for training, and unhelpful when evaluating the model against. </b>
  2) There are only 7 in total, and only one female suspect sample.
 
* We combine all HCV+ samples (Hepatitis, Fibrosis, and Cirrhosis) into a single HCV+ group. The question we're trying to answer with this model is whether a sample comes from a HCV+ or HCV- person, we're not looking to say what disease state they are in. Combining samples also will help ensure we have enough HCV+ samples to have a good distribution for both training and testing data sets.

After making these modifications we have the following distributions:

<img width="842" height="545" alt="output_13_1" src="https://github.com/user-attachments/assets/ae3847fc-2347-4e95-b486-c5ce7c51845c" />

<img width="1005" height="622" alt="output_21_1" src="https://github.com/user-attachments/assets/b6ebfcbe-74a1-4be3-8bc4-567d4350e5c2" />

Now, there is one last step before we can begin the analysis. We need to turn Sex into a categorical variable in order to use it in a decision tree. Thankfully the pandas get_dummies() function is able to do this for us. It split the column into two - Sex_f, and Sex_m, both are boolean columns, with a value of 1 to denote whether the individual the associated sex. All of this manipulation results in a dataframe consisting of the following information:

```python
<class 'pandas.core.frame.DataFrame'>
Index: 608 entries, 0 to 614
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Category  608 non-null    object 
 1   Age       608 non-null    int64  
 2   ALB       607 non-null    float64
 3   ALP       590 non-null    float64
 4   AST       608 non-null    float64
 5   BIL       608 non-null    float64
 6   CHE       608 non-null    float64
 7   CHOL      598 non-null    float64
 8   CREA      608 non-null    float64
 9   CGT       608 non-null    float64
 10  PROT      607 non-null    float64
 11  ALT       607 non-null    float64
 12  Sex_f     608 non-null    bool   
 13  Sex_m     608 non-null    bool   
dtypes: bool(2), float64(10), int64(1), object(1)
memory usage: 62.9+ KB
```

The last thing we need to do is separate out 'Category' into it's own separate dataframe again.

# Decision Tree
Using sklearn's model_selection.train_test_split we first separate our data into a training set and a testing set. Each set includes an X dataframe with our features, and y dataframe with the answers (in this case the 'Category' column). Note that we're setting the random state for reproducibility. If you run this code you should see the same answers as described here.

```python
# Split data into a training set and a testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 105)
```

Next we import DecisionTreeClassifier, and train our model.
```python

# Time to train the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)
```
 This results in the following decision tree:

<img width="794" height="964" alt="output_31_1" src="https://github.com/user-attachments/assets/a4da8c42-5982-412d-adf4-a08a9679fec6" />

Each node is broken up into the following information  </br>
<b> Feature Name and Decision </b> - The name of the feature used and what ther cutoff points are. </br>
<b> samples </b> - The number of samples from our training set that were passed to the node  </br>
<b> value </b> - The distribution of HCV- and HCV+ samples that were passed to this node  </br>
<b> class </b> = Whether the majority of samples passed to this node were HCV+ or HCV-  </br>

When we use this decision tree on the test set we get the following report:

```python
              precision    recall  f1-score   support

 Blood Donor       0.99      0.98      0.98       161
   Hepatitis       0.87      0.91      0.89        22

    accuracy                           0.97       183
   macro avg       0.93      0.95      0.94       183
weighted avg       0.97      0.97      0.97       183
```

And the following confusion matrix / sensitivity and specificity.

<img width="559" height="453" alt="output_37_1" src="https://github.com/user-attachments/assets/6b0015ab-7105-4471-afcc-d77a88513894" />

* Out of the 183 samples in the test dataset 161 were HCV - and 22 were HCV +
* The model accurately predicted that 158/161 negative samples, and 20/22 positive samples.
* However, there were 2 false negatives, and 3 false positives.

# Random Forest

Random forests are less straightforward. Instead of a single decision tree they create an ensemble of decision trees, where each tree is built using a random subset of data and features. This makes them more robust and accurate due to averaging predications from many trees. Random forests are also and less prone to overfitting. We can also use the same split data sets that were used previously for the decision tree.

``` python

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)

```

We get the following results using the random forest model: 

```python
              precision    recall  f1-score   support

 Blood Donor       0.99      1.00      0.99       161
   Hepatitis       1.00      0.91      0.95        22

    accuracy                           0.99       183
   macro avg       0.99      0.95      0.97       183
weighted avg       0.99      0.99      0.99       183

```

<img width="559" height="453" alt="output_46_1" src="https://github.com/user-attachments/assets/3edb1421-7e42-44a0-8148-3d81017ebf03" />

Notice that our accuracy has increased.
* This model correctly predicted all negative samples.
* However there are still 2 false positive samples.

# Random Forest Model Features
Since a Random Forest model runs ensembles of decision trees with randomized samples and featured we can get aggregate data on what features are the most important. We'll do this in two ways:

## Permutation importantance
Permutation importance measures the contribution of each feature to the models statistical performance on a given dataset. It involves randomly shuffling the values of a single feature then observing how that impacts the model's performance. Particularly the f1 score.
<img width="630" height="470" alt="output_51_0" src="https://github.com/user-attachments/assets/b5123f21-9811-4eeb-b400-1d2e70388f24" />

ALP and AST are the most important features according to analysis based on permutations
* AST stands for aspartate aminotransferase. It's an enzyme that speeds up chemical reactions in the body, it's found primarily in organs and tissues with relatively little found in blood. However, AST is released into the bloodstream following liver damage, or damage to AST producing cells. It makes sense that AST would be importnant here.
* ALP stands for alkaline phosphatase, another enzyme found in the liver (as well as some other tissues and bone). It's believed to be involved in multiple processes. While it doesn't directly indicate liver damage or inflammation elevated levels signal a problem with bile flow or pressure within the liver.

## gini index
The gini index evaluates the quality of a split in a decision tree or the impurity. Ideally, all HCV- samples would be separated into one group and all HCV+ samples into another group creating two pure groups at that decision point. Obviously, that is a dream scenerio. So the gini index is a measure of impurity, so lower indexes are better. The graph below looks at the mean decrease in impurity by a given feature. This is not the exact gini index but another way to look at it. Values correlate with how much a feature lowers the impurity. Higher values means less impurity, and thus represent more importance features.

<img width="630" height="470" alt="output_56_0" src="https://github.com/user-attachments/assets/96cc926f-6345-4b01-a350-9441b395ba7a" />

Once again ALP and AST are the most important features, although their positions have switched.

# Gradient Boosting

Gradient Boosting creates strong predictive models by sequentally combining multiple weaker models, like shallow decision trees. It works iteratively by adding new models which focus on correcting errors made by the existing ensemble. The algorithm works to minimize a loss function, in this case cross entropy since it's being used for classification. It has a few parameters we'll cover more in depth a bit later. 

## Preparation
However, first we need to do some data manipulation. The dataset contains a few Nan values, and traditional Gradient Boosting is unable to work with missing values. There are two options: 
</br>
1) A histogram boosting gradient classification is able to handle missing values natively. It learns how to handle them by determining at each node whether samples with missing values should go left or right based on the potential gain. This is the easier of the two options, as it doesn't actually require any data manipulation!
</br>
2) Imputing data to fill in missing values or remove rows containing missing values. This can be done in the following ways, though each has potential drawbacks:
   </br>
   * Removing rows with any missing values - this will lower our sample size
   </br>
   * Using the mean or median for each category of missing values - Since we have an unbalanced dataset, the average will skew more towards the healthy blood donor values, assuming that there is a difference between the groups. While we could separate groups and use mean or median values based on known disease status- such a technique is not possible for samples with unknown disease states (which are what this model is meant to be used for. That's a bit of a Catch-22!)
   </br>
   * Assigning values using a regression method based on the values that are present for that row - There are a few regression methods that can be employed for these predictions. And given our dataset this is the best option.   
</br>
   * Making missing values their own classification for prediction. For example there could be a node that sorts the datapoints based on whether or not they is missing ALT results. - This method makes little sense for our use case, as missing test results likely won't impact whether or not someone is predicted to have Hepatitis, it just means there was an issue during data collection or with the lab results.
 </br>

For simplicity, first we'll use the histogram gradient classification.

``` python

from sklearn.ensemble import HistGradientBoostingClassifier

```

First we load the package, then as with the random forest classification we create a class instance and fit our training data. 

``` python

hgbr = HistGradientBoostingClassifier()
hgbr.fit(X_train, y_train)

```

To evaluate this model we're actually going to use the cross_val_score function from sklearn, you'll see why soon. This function performs a cross-validation process to evaluate and estimate a model's performance on unseen data. It does this by splitting the dataset into multiple folds, trains the model on a subset of these folds and then evaluates it on the remaining fold - kind of like what we did by separating the dataset with train_test_split. However, this is done for each fold and scores from each iteration are returned.

``` python

from sklearn.model_selection import GridSearchCV, cross_val_score
cross_val_score(hgbr, X_train, y_train, cv =3, n_jobs =1).mean()

```

``` python

0.9835347784104153

```

98.3% Is pretty good, but there is room for improvement.

``` python
 
predictions = hgbr.predict(X_test)
print(classification_report(y_test, predictions))

```

```python

              precision    recall  f1-score   support

 Blood Donor       0.99      1.00      1.00       161
   Hepatitis       1.00      0.95      0.98        22

    accuracy                           0.99       183
   macro avg       1.00      0.98      0.99       183
weighted avg       0.99      0.99      0.99       183

```
This model is already an improvement over the Random Forest Classification. It correctly predicted ass negative samples, and only had 1 false negative prediction. This is why we needed to use the cross_val_score function to evaluate performance, as it is more robust to small tweaks and improvements. 
<img width="559" height="453" alt="output_31_1" src="https://github.com/user-attachments/assets/6a01bdc6-9880-40f7-917c-efdafc39f08a" />

Everything up until now has been similar to the Random Forest Classification method, here is where it gets a bit different. As the name suggests Gradient Boosting involves some boosting. This method allows a few parameter changes for optimization. The parameters are as follows:
</br>
* Learning rate
* Max iterations
* Max Depth
</br>

To optimize these parameters we'll create a grid of different values to test for each parameter. 
``` python
param_grid = {
    'max_iter': [10, 50, 100, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
    'max_depth': [3, 5, 7, 9]
}
```
Next we use GridSearchCV to explore all parameter combinations with our training data.

```python

hgbr2 = GridSearchCV(hgbr, param_grid, cv=3, n_jobs=-1)
hgbr2.fit(X_train, y_train)

```
We can then look at which combination of parameters are optimal.
```python
hgbr2.best_params_
```

```python

{'learning_rate': 0.1, 'max_depth': 3, 'max_iter': 50}

```
As well as what score the model gives when using these parameters.
```python

hgbr2.best_score_

```

```python

0.9882462624446443

```
We went from a score 98.35% to 98.8%. There wasn't much room to improve. 

```python

final_hgbr =  HistGradientBoostingClassifier(learning_rate = 0.1, max_depth = 3, max_iter = 100)
final_hgbr = final_hgbr.fit(X_train, y_train)
predictions = final_hgbr.predict(X_test)
print(classification_report(y_test, predictions))

```

```python

             precision    recall  f1-score   support

 Blood Donor       0.98      1.00      0.99       159
   Hepatitis       1.00      0.88      0.93        24

    accuracy                           0.98       183
   macro avg       0.99      0.94      0.96       183
weighted avg       0.98      0.98      0.98       183

```
This 'improved' model has identical results to it's predicessor when using our training set. Once again, there is a single false positive prediction.


<img width="559" height="453" alt="output_35_1" src="https://github.com/user-attachments/assets/153481e0-8974-45d2-9d64-9579052305e6" />


