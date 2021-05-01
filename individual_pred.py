import databaseconfig as cfg
import functions as fn
import main as mn
import warnings
warnings.filterwarnings('ignore')

user_list, predA, predB = fn.individual_pred(mn.data, mn.xgb_reg)
alternative_eval = fn.alt_eval(user_list, predA, predB)
alternative_eval = fn.alt_eval(user_list, predA, predB)
