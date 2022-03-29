# [House Prices - Advanced Regression Techniquess](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Project for [Fundamentals of Data Science](https://sites.google.com/di.uniroma1.it/fds-2021-2022/home?authuser=0) course (A.Y. 2018/2019).

Full details are available in the [report](https://github.com/icannistraci/kaggle-house-prices/report2019.pdf).

Score: _0.11309_ (**top 3%** of the leaderboard).

## Data tidying​
1. Removed outliers
2. Replaced the following values of Exterior2nd with the correct value [’CmentBd’: ’CemntBd’, ’Wd Shng’:
’WdShing’, ’Brk Cmn’: ’BrkComm’].
3. Replace nan values of the following categorical features values with the correct value (e.g. for Alley I used ’NoAlleyAccess’) since in these cases, nan doesn’t mean missing value [Alley, BsmtQual, Bsmt- Cond, BsmtExposure, BsmtFinType1, BsmtFinType2, Fireplaces, FireplaceQu, GarageType].

## Features engineering​
I decided to merge the train set and the test set and to analyze, starting from the data description file, every single feature to understand what was the best way to treat them.

_Missing values:_
1. Most of the nan values were wrong values that I previously adjusted (explained in Data tidying).
2. Some features like MSZoning, LotFrontage, BsmtFullBath, and BsmtHalfBath were replaced with mode and mean.
3. All the other values were replaced using different criteria based on the specific feature. In most cases the case I used the most frequent value but for example, for the set of features about Garage, I used to put them together and check if the nan values of them means that they have no garage, so in these cases I filled the missing field with 0 (or with the corrective categorical value).

_New features:_
1. I created many boolean features (0,1). For example the feature ’2ndFlrSF’ if is equal to 1, the corrispective boolean feature ’Has2ndFlr’ will have value 1, otherwise 0. [HasAlley, HasRegularShape, HasCondi- tion2, IsRemodeled, HasMoreMaterials, HasMasVnr, HasMultipleBsmtType, HasBsmt, Has2ndFlr, Has- Fireplaces, HasGarage, HasPavedDrive, HasWoodDeck, HasOpenPorch, HasEnclPorch, ’Has3SsnPorch’, ’HasScreenPorch’, HasPool].
2. I created new numerical features like TotalSF that represents total floor square feet [TotalSF, Foundati- onQual, FunctionalQual]
3. I created numerical features for the one that had both a value for quality and condition. I did the mean between them [OverallQualCondMean, ExterQualCondMean, BsmtQualCondMean, GarageQualCondMean].

_Categorical features:_
1. I transformed features with only 2 possible values into boolean (0,1) [Street, CentralAir]
2. I transformed features that represent a ’vote’ into numerical features (ordinal) where every value had  a corresponding integer that is the same for all these features [Po: 1, Fa: 2, TA: 3, Gd: 4, Ex: 5] and
[Unf: 1, LwQ: 2, Rec: 3, BLQ: 4, ALQ: 5, GLQ: 6] where increasing value means better.
3. Other categorical features were transformed using one hot encoding (pandas getDummies). Then I removed values ’Metal’, ’Membran’ and ’Roll’ of ’Roof Matl’, ’Mix’ of ’Electrical’ and ’OthW’ and ’Floor’ of ’Heating’ feature since there are no rows with that values in the test set.

_Deleted features:_
5. I deleted the following features: LotShape because I replaced it with HasRegularShape and PavedDrive because I replaced it with HasPavedDrive, Utilities since 2908 rows over 2909 had the same value (All- Pub) and PoolQC since 2902 rows over 2909 had null values.


## Training​
1. Performing the predictions on the price log to normalize the exponential distribution of the house's price.
2. Weighted mean of the predictions from the following models: RidgeCV, LassoCV, ElasticNetCV, GradientBoostingRegressor, BayesianRidge, StackingCVRegressor. The predictors are all the previous (without the cross-validation) in the stacking regressor, and the meta predictor is a Lasso model. The cross-validation is a 20-folds cross-validation, and we perform a refit on all the data after selecting the best parameters.
3. Normalizing the outliers (price too high or too low) of the predicted price
4. Rounding the predicted price to multiples of 1000
