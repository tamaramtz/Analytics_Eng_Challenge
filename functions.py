## Import Libraries
import numpy as np
import databaseconfig as cfg
from scipy.stats import norm, skew
import sklearn.metrics as metrics
# ML Libraries
import xgboost as XGB
import warnings

warnings.filterwarnings('ignore')


# -- -------------------------------------------------------------- FUNCTION: Fix Skewness -- #
# -- ------------------------------------------------------------------------------------ -- #
# -- Fix the skewness of every feature by normalizing them
def fix_skewness(data):
    """
    :param data: DataFrame
    :return: DataFrame normalized
    """
    numeric_feats = data.dtypes[data.dtypes != 'object'].index
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    for feature in high_skew.index:
        data[feature] = np.log1p(data[feature])
    return data


# -- -------------------------------------------------------------- FUNCTION: Split Data -- #
# -- ------------------------------------------------------------------------------------ -- #
# -- Split the data into train and test
def split_data(data):
    """
    :param data: DataFrame
    :return: Train and test variables
    """
    np.random.seed(2018)
    train = np.random.choice([True, False], data.shape[0], replace=True, p=[0.8, 0.2])
    listings_train = data.iloc[train, :]
    listings_test = data.iloc[~train, :]
    train_cols = ['OverallQual', 'FullBath', 'YearBuilt', 'GrLivArea', 'GarageCars']
    target_col = 'SalePrice'

    x_train = listings_train[train_cols].values
    x_test = listings_test[train_cols].values
    y_train = listings_train[target_col].values
    y_test = listings_test[target_col].values
    return x_train, x_test, y_train, y_test


# -- -------------------------------------------------------------- FUNCTION: Model -- #
# -- ------------------------------------------------------------------------------------ -- #
# -- Create ML model
def model(x_train, y_train):
    """
    :param x_train: Array of train observations
    :param y_train:  Array of train target variable
    :return: ML model
    """
    xgb_reg = XGB.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3)
    xgb_reg.fit(x_train, y_train)
    return xgb_reg


# -- -------------------------------------------------------------- FUNCTION: Model Performance -- #
# -- ------------------------------------------------------------------------------------ -- #
# -- Measure ML model performance
def model_performance(x_train, x_test, y_train, y_test, xgb_reg):
    """
    :param x_train: Array of train observations
    :param x_test: Array of train target variable
    :param y_train: Array of test observations
    :param y_test: Array of test target variable
    :param xgb_reg: Model
    :return: Performance metrics of the model
    """
    y_pred = xgb_reg.predict(x_train)
    print('Train Data Model Performance')
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.expm1(y_train), np.expm1(y_pred))))
    print('Mean Absolute Error:', metrics.mean_absolute_error(np.expm1(y_train), np.expm1(y_pred)))
    print('R^2:', metrics.r2_score(np.expm1(y_train), np.expm1(y_pred)), '\n')

    y_pred = xgb_reg.predict(x_test)
    print('Test Data Model Performance')
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.expm1(y_test), np.expm1(y_pred))))
    print('Mean Absolute Error:', metrics.mean_absolute_error(np.expm1(y_test), np.expm1(y_pred)))
    print('R^2:', metrics.r2_score(np.expm1(y_test), np.expm1(y_pred)))


# -- -------------------------------------------------------------- FUNCTION: Individual Predictions -- #
# -- ------------------------------------------------------------------------------------ -- #
# -- Individual predictions
def individual_pred(data, xgb_reg):
    """
    :param data: DataFrame
    :param xgb_reg: Model
    :return: Individual predictions
    """
    input_string = input('Enter id or ids separated by space: ')
    print("\n")
    user_list = input_string.split()

    predA = []
    predB = []

    # convert each item to int type
    for i in range(len(user_list)):
        # convert each item to int type
        user_list[i] = int(user_list[i])
    for i in user_list:
        id_input = data[data['Id'] == i]
        input_a = id_input.drop(['Id', 'SalePrice'], axis=1)
        price_pred = xgb_reg.predict(input_a.values)
        method_a = np.floor(np.expm1(price_pred[0]))
        predA.append(method_a)

        mu = np.mean(data['SalePrice'])
        sigma = np.std(data['SalePrice'])
        method_b = np.floor(np.expm1(np.random.normal(mu, sigma)))
        predB.append(method_b)
    print('The predicted price for id ', user_list, ' with method A is: ', predA)
    print('The predicted price for id ', user_list, ' with method B is: ', predB)

    return user_list, predA, predB


# -- -------------------------------------------------------------- FUNCTION: Individual Predictions Evaluation -- #
# -- ------------------------------------------------------------------------------------ -- #
# -- Alternative Evaluation Method
def alt_eval(user_list, predA, predB):
    """
    :param train: Original DataFrame
    :param user_list: Input
    :param predA: Prediction with Method A
    :param predB: Prediction with Method B
    :return: Percentage of Bad predictions
    """
    good_predA, good_predB = 0, 0
    id_input = []
    for i in user_list:
        price = cfg.train['SalePrice'][cfg.train['Id'] == i].to_list()
        id_input.append(np.expm1(price))
    for j in range(0, len(id_input)):
        if np.abs(id_input[j] - predA[j]) < 10000:
            good_predA += 1
        if np.abs(id_input[j] - predB[j]) < 10000:
            good_predB += 1
    bad_predA = (np.abs(len(user_list) - good_predA)) / len(user_list)
    bad_predB = (np.abs(len(user_list) - good_predB)) / len(user_list)
    print('Bad prediction percentage with method A', bad_predA * 100, '%')
    print('Bad prediction percentage with method B', bad_predB * 100, '%')
    return id_input
