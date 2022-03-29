import numpy as np
import random
import datetime

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge, BayesianRidge
from sklearn.model_selection import KFold
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def mlp_regression(x, y, RANDOM_STATE, kfold):
    # Grid search
    k_best = 0
    M = 100000000
    RMSE_best = M
    R2_best = -1
    y_pred_best = np.zeros_like(y)

    # 5 Layer with (from 10 to 90) neurons
    poss_numlayer = []
    num_of_couple = 3
    for num in range(0, num_of_couple):
        pair = 5, random.randint(5, 30)
        poss_numlayer.append(pair)
    print('poss_numlayer', poss_numlayer)

    k = 0
    # poss_actfun = ['identity', 'logistic', 'tanh', 'relu']
    poss_actfun = ['relu']
    for numlayer in poss_numlayer:
        for actfun in poss_actfun:
            y_pred = np.zeros_like(y)
            for train_index, test_index in KFold.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                reg = MLP(hidden_layer_sizes=numlayer, activation=actfun, solver='lbfgs',
                          random_state=RANDOM_STATE, tol=10e-6, max_iter=1000)
                reg.fit(x_train, y_train)

                y_pred[test_index] = reg.predict(x_test)

            RMSE = mean_squared_error(np.log(y, where=y > 0), np.log(y_pred, where=y_pred > 0))
            R2 = r2_score(y, y_pred)

            print('MLP ', 'k=', k, "layer/neur=", numlayer, "ActFun= ", actfun, "Solver=lbfgs",
                  'R2=', R2, "RMSE=", RMSE)

            if R2 > R2_best:
                R2_best = R2
                y_pred_best = y_pred
                k_best = k
                numlayer_best = numlayer
                actfun_best = actfun

            k += 1

    return y_pred_best, R2_best, y, k_best, numlayer_best, actfun_best


def stacked_regression(X_train, Y_train, X_test, RANDOM_STATE, kfold):
    print('Start stacked regression..', datetime.datetime.now())

    lasso = Lasso(alpha=0.00035, random_state=RANDOM_STATE, max_iter=50000)

    gradientboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                              max_depth=4, max_features='sqrt',
                                              min_samples_leaf=15, min_samples_split=50,
                                              loss='huber', random_state=5)

    elasticnet = ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3)

    ridge = Ridge(alpha=9.0, fit_intercept=True)

    bayesian = BayesianRidge(fit_intercept=True, verbose=True, n_iter=10000)

    rf = RandomForestRegressor(random_state=RANDOM_STATE)

    stack = StackingCVRegressor(regressors=(lasso, gradientboost, elasticnet, ridge, bayesian, rf),
                                meta_regressor=rf,
                                use_features_in_secondary=True,
                                random_state=RANDOM_STATE,
                                cv=kfold)

    stack.fit(X_train, Y_train)
    x_test_predict = stack.predict(X_test)
    print('End stacked regression..', datetime.datetime.now())

    return x_test_predict


def xgb_regression(X_train, Y_train, X_test, RANDOM_STATE, kfold):
    print('Start xgb regression..', datetime.datetime.now())

    xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                           max_depth=3, min_child_weight=0,
                           gamma=0, subsample=0.7,
                           colsample_bytree=0.7,
                           objective='reg:squarederror', nthread=-1,
                           scale_pos_weight=1, seed=27,
                           reg_alpha=0.00006,
                           random_state=RANDOM_STATE,
                           cv=kfold)

    xgboost.fit(X_train, Y_train)
    x_test_predict = xgboost.predict(X_test)

    print('End xgb regression..', datetime.datetime.now())

    return x_test_predict
