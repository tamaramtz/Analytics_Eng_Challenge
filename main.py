import databaseconfig as cfg
import functions as fn
import warnings
warnings.filterwarnings('ignore')

# %% Data extraction
data = cfg.train
# %% Data Processing
data = fn.fix_skewness(data)
# %% Model training
# Split Dataset into train and test
x_train, x_test, y_train, y_test = fn.split_data(data)
print('X train shape', x_train.shape, 'X test shape', x_test.shape,
      'Y train shape', y_train.shape, 'Y train shape', y_test.shape)
# Create ML Model
xgb_reg = fn.model(x_train, y_train)
# %% Measure Model Performance
performance = fn.model_performance(x_train, x_test, y_train, y_test, xgb_reg)
