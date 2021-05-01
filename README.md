# Analytics_Eng_Challenge
The objective of this challenge is to assess the ability to develop a simple ML repo, following good coding practices and considering an appropriate structure.

### ML training pipeline
Created a ML training pipeline which contains the following steps:
+ Data extraction (from the kaggle website)
+ Data processing (feature computation)
+ Model training
+ Model evaluation

### Generate individual predictions
Created a script which generates individual predictions of your model.
It works in the following way:
+ input: you run the script from a terminal, passing as an argument the Id of a house
(the same Id found in the raw dataset).
+ output: the script returns to stdout the prediction of the price for that house
+ internal process: it gets the raw data, computes and processes the features required by the
model for that Id
+ loads model
+ feeds model with the processed features and obtains a prediction

### Implemented an experiment on individual predictions
**Implementing an alternative evaluation method** 
“Method A” is the prediction method implemented in the previous section. For a
random 50% of individual predictions, it will generate a prediction using “method B”,
which we define now.
Method B simply consists of predicting a random value for the SalePrice variable. However,
this random value must be drawn from the distribution of the SalePrice variable. 
**Designing an experiment**
Compare the accuracy of methods A and B. Because of commercial
reasons, a prediction which is less than 10,000 USD away from the actual SalePrice is
considered a good prediction. Otherwise, predictions are considered bad predictions.
