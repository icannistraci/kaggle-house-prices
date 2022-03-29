import pandas as pd
import numpy as np
import UtilitiesFunctions as UtilFunct
import regression_models as Models
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler


def warning_ignore(*args, **kwargs):
    pass


warnings.warn = warning_ignore

# initialize things that I'll need
feat_to_drop = []
numeric_feat = []
one_hot_enc_feat = []
# use it to understand if the model will work
# better or not with new features
new_feat = []

# load data from csv
x_train = pd.read_csv('../data/train.csv', delimiter=',')
x_test = pd.read_csv('../data/test.csv', delimiter=',')

# removing outliers as suggested from community
out = [30, 88, 462, 631, 1322]
x_train = x_train.drop(x_train.index[out])
x_train.drop(x_train[(x_train['OverallQual'] < 5) & (x_train['SalePrice'] > 200000)].index, inplace=True)
x_train.drop(x_train[(x_train['GrLivArea'] > 4000) & (x_train['SalePrice'] < 300000)].index, inplace=True)
x_train.reset_index(drop=True, inplace=True)

# get SalePrice (y_train)
y_train = x_train['SalePrice']
x_train.drop('SalePrice', inplace=True, axis=1)

# before drop id check features with 99% of equal value
equal_val_train = UtilFunct.check_equal_value(x_train, 0.99)
equal_val_test = UtilFunct.check_equal_value(x_test, 0.99)
# train = Street, Utilities, PoolArea
# test = Street, Utilities, Heating, 3SsnPorch, PoolArea

# drop Ids
id_train = x_train['Id']
x_train.drop('Id', inplace=True, axis=1)
id_test = x_test['Id']
x_test.drop('Id', inplace=True, axis=1)

# merge train and test
x_merged = pd.concat([x_train, x_test], sort=False).reset_index(drop=True)

# START analyzing features

# MSSubClass: Identifies the type of dwelling involved in the sale.
# 0/2909 null values
# print('MSSubClass: ', x_merged['MSSubClass'].isnull().sum())
# this is numeric but seems categorical
one_hot_enc_feat.append('MSSubClass')

# MSZoning: Identifies the general zoning classification of the sale.
# 4/2909 null values
# print('MSZoning: ', x_merged['MSZoning'].isnull().sum())
# group by MSSubClass and fill in missing value by the mode
x_merged['MSZoning'] = x_merged.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# ({'RL': 2261, 'RM': 462, 'FV': 139, 'RH': 26, 'C (all)': 23})
# print(Counter(x_merged['MSZoning']))
one_hot_enc_feat.append('MSZoning')

# LotFrontage: Linear feet of street connected to property
# 485/2909 null values
# print('LotFrontage: ', x_merged['LotFrontage'].isnull().sum())
# group by Neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
x_merged['LotFrontage'] = x_merged.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
numeric_feat.append('LotFrontage')

# LotArea: Lot size in square feet
# 0/2909 null values
# print('LotArea: ', x_merged['LotArea'].isnull().sum())
numeric_feat.append('LotArea')

# Street: Type of road access to property
# 4/2909 null values
# ({'Pave': 2899, 'Grvl': 12})
# print(Counter(x_merged['Street']))
# decide if change into 0,1
# one_hot_enc_feat.append('Street')
x_merged['Street'].replace({'Grvl': 0, 'Pave': 1}, inplace=True)

# Alley: Type of alley access to property
# 2714/2909 null values -> NO ALLEY ACCESS
# print('Alley: ', x_merged['Alley'].isnull().sum())
x_merged['Alley'] = x_merged['Alley'].fillna('NoAlleyAccess')
# ({'NoAlleyAccess': 2714, 'Grvl': 120, 'Pave': 77})
# print(Counter(x_merged['Alley']))
one_hot_enc_feat.append('Alley')

# ADD NEW FEATURE HasAlley 0,1
x_merged['HasAlley'] = np.where(x_merged['Alley'] != 'NoAlleyAccess', 1, 0)
new_feat.append('HasAlley')

# LotShape: General shape of property
# 0/2909 null values
# ({'Reg': 1857, 'IR1': 963, 'IR2': 76, 'IR3': 15})
# print(Counter(x_merged['LotShape']))
# CHANGE IT INTO: HasRegularShape 0,1
x_merged['HasRegularShape'] = np.where(x_merged['LotShape'] != 'Reg', 0, 1)
feat_to_drop.append('LotShape')
new_feat.append('HasRegularShape')

# LandContour: Flatness of the property
# 0/2909 null values
# ({'Lvl': 2617, 'HLS': 120, 'Bnk': 115, 'Low': 59})
# print(Counter(x_merged['LandContour']))
one_hot_enc_feat.append('LandContour')

# Utilities: Type of utilities available
# 2/2909 null values
# 2908/2909 AllPub
# ({'AllPub': 2908, nan: 2, 'NoSeWa': 1})
# print(Counter(x_merged['Utilities']))
feat_to_drop.append('Utilities')

# LotConfig: Lot configuration
# 0/2909 null values
# ({'Inside': 2128, 'Corner': 509, 'CulDSac': 175, 'FR2': 85, 'FR3': 14})
# print(Counter(x_merged['LotConfig']))
one_hot_enc_feat.append('LotConfig')

# LandSlope: Slope of property
# 0/2909 null values
# ({'Gtl': 2771, 'Mod': 124, 'Sev': 16})
# print(Counter(x_merged['LandSlope']))
one_hot_enc_feat.append('LandSlope')

# Neighborhood: Physical locations within Ames city limits
# 0/2909 null values
# print('Neighborhood: ', x_merged['Neighborhood'].isnull().sum())
one_hot_enc_feat.append('Neighborhood')

# Condition1: Proximity to various conditions
# 0/2909 null values
# ({'Norm': 2507, 'Feedr': 161, 'Artery': 92, 'RRAn': 50, 'PosN': 38, 'RRAe': 28, 'PosA': 20, 'RRNn': 9, 'RRNe': 6})
# print(Counter(x_merged['Condition1']))
one_hot_enc_feat.append('Condition1')

# Condition2: Proximity to various conditions (if more than one is present)
# 0/2909 null values
# ({'Norm': 2883, 'Feedr': 12, 'Artery': 5, 'PosA': 4, 'PosN': 3, 'RRNn': 2, 'RRAn': 1, 'RRAe': 1})
# print(Counter(x_merged['Condition2']))
one_hot_enc_feat.append('Condition2')
# ADD NEW FEATURE HasCondition2 0,1
x_merged['HasCondition2'] = np.where(x_merged['Condition1'] != x_merged['Condition2'], 1, 0)
# ({0: 2518, 1: 393})
# print(Counter(x_merged['HasCondition2']))
new_feat.append('HasCondition2')

# BldgType: Type of dwelling
# 0/2909 null values
# ({'1Fam': 2418, 'TwnhsE': 227, 'Duplex': 109, 'Twnhs': 95, '2fmCon': 62})
# print(Counter(x_merged['BldgType']))
one_hot_enc_feat.append('BldgType')

# HouseStyle: Style of dwelling
# 0/2909 null values
# ({'1Story': 1468, '2Story': 868, '1.5Fin': 313, 'SLvl': 128, 'SFoyer': 83, '2.5Unf': 24, '1.5Unf': 19, '2.5Fin': 8})
# print(Counter(x_merged['HouseStyle']))
one_hot_enc_feat.append('HouseStyle')
# Understand what split foyer is and if I can change it into new features

# OverallQual: Rates the overall material and finish of the house
# 0/2909 null values
# print('OverallQual: ', x_merged['OverallQual'].isnull().sum())
numeric_feat.append('OverallQual')
# OverallCond: Rates the overall condition of the house
# 0/2909 null values
# print('OverallCond: ', x_merged['OverallCond'].isnull().sum())
numeric_feat.append('OverallCond')

# ADD NEW FEATURE OverallQualCondMean
x_merged['OverallQualCondMean'] = round(x_merged[['OverallQual', 'OverallCond']].mean(axis=1))
new_feat.append('OverallQualCondMean')
numeric_feat.append('OverallQualCondMean')

# YearBuilt: Original construction date
# 0/2909 null values
# print('YearBuilt: ', x_merged['YearBuilt'].isnull().sum())
numeric_feat.append('YearBuilt')

# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
# 0/2909 null values
# print('YearRemodAdd: ', x_merged['YearRemodAdd'].isnull().sum())
numeric_feat.append('YearRemodAdd')

# ADD NEW FEATURE IsRemodeled
x_merged['IsRemodeled'] = np.where(x_merged['YearBuilt'] != x_merged['YearRemodAdd'], 1, 0)
new_feat.append('IsRemodeled')

# RoofStyle: Type of roof
# 0/2909 null values
# ({'Gable': 2307, 'Hip': 548, 'Gambrel': 21, 'Flat': 19, 'Mansard': 11, 'Shed': 5})
# print(Counter(x_merged['RoofStyle']))
one_hot_enc_feat.append('RoofStyle')

# RoofMatl: Roof material
# 0/2909 null values
# ({'CompShg': 2870, 'Tar&Grv': 22, 'WdShake': 9, 'WdShngl': 7, 'Metal': 1, 'Membran': 1, 'Roll': 1})
# print(Counter(x_merged['RoofMatl']))
roof_matl_check = x_merged['RoofMatl'][(x_merged['RoofMatl'] == 'Metal') | (x_merged['RoofMatl'] == 'Membran')
                                       | (x_merged['RoofMatl'] == 'Roll')]
one_hot_enc_feat.append('RoofMatl')

# Exterior1st: Exterior covering on house
# 1/2909 null values
# ({'VinylSd': 1024, 'MetalSd': 449, 'HdBoard': 441, 'Wd Sdng': 411, 'Plywood': 219, 'CemntBd': 125, 'BrkFace': 86,
# 'WdShing': 56, 'AsbShng': 44, 'Stucco': 42, 'BrkComm': 6, 'AsphShn': 2, 'Stone': 2, 'CBlock': 2, 'ImStucc': 1,
# nan: 1})
# print(Counter(x_merged['Exterior1st']))
# fill missing value with VinylSd, id: 2143
x_merged.loc[2143, 'Exterior1st'] = 'VinylSd'
one_hot_enc_feat.append('Exterior1st')
# Exterior2nd: Exterior covering on house (if more than one material)
# 1/2909 null values
# ({'VinylSd': 1013, 'MetalSd': 446, 'HdBoard': 405, 'Wd Sdng': 391, 'Plywood': 268, 'CmentBd': 125, 'Wd Shng': 81,
# 'BrkFace': 46, 'Stucco': 46, 'AsbShng': 38, 'Brk Cmn': 22, 'ImStucc': 15, 'Stone': 6, 'AsphShn': 4, 'CBlock': 3,
# 'Other': 1, nan: 1})
# print(Counter(x_merged['Exterior2nd']))
# adjust values
x_merged['Exterior2nd'].replace({'CmentBd': 'CemntBd', 'Wd Shng': 'WdShing', 'Brk Cmn': 'BrkComm'}, inplace=True)
# fill missing value by same value of Exterior1st, id: 2143
x_merged.loc[2143, 'Exterior2nd'] = 'VinylSd'
one_hot_enc_feat.append('Exterior2nd')

# ADD NEW FEATURE HasCondition2 0,1
x_merged['HasMoreMaterials'] = np.where(x_merged['Exterior1st'] != x_merged['Exterior2nd'], 1, 0)
# ({0: 2643, 1: 268})
# print(Counter(x_merged['HasMoreMaterials']))
new_feat.append('HasMoreMaterials')

# MasVnrType: Masonry veneer type
# 24/2909 null values
# ({'None': 1737, 'BrkFace': 879, 'Stone': 246, 'BrkCmn': 25, nan: 24})
# print(Counter(x_merged['MasVnrType']))
x_merged['MasVnrType'] = x_merged['MasVnrType'].fillna(0)

# MasVnrArea: Masonry veneer area in square feet
# 23/2909 null values
# print('MasVnrArea: ', x_merged['MasVnrArea'].isnull().sum())
# to understand how to fill nan value
masonry_null = (x_merged[['MasVnrType', 'MasVnrArea']][x_merged[['MasVnrType', 'MasVnrArea']]["MasVnrType"] == 0])
x_merged['MasVnrArea'] = x_merged['MasVnrArea'].fillna(0)
# since record 2602 has area = 198 then I replaced it with BrkFace
x_merged.loc[2602, 'MasVnrType'] = 'BrkFace'
one_hot_enc_feat.append('MasVnrType')
numeric_feat.append('MasVnrArea')

# ADD NEW FEATURE: HasMasVnr 0,1
x_merged['HasMasVnr'] = np.where(x_merged['MasVnrType'] == 'None', 0, 1)
new_feat.append('HasMasVnr')

# ExterQual: Evaluates the quality of the material on the exterior
# ExterCond: Evaluates the present condition of the material on the exterior
# 0/2909 null values
# ({'TA': 1795, 'Gd': 977, 'Ex': 105, 'Fa': 34})
# print(Counter(x_merged['ExterQual']))
# CHANGE IT into numerical: increasing values means better
x_merged['ExterQual'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
numeric_feat.append('ExterQual')

# 0/2909 null values
# ({'TA': 2532, 'Gd': 299, 'Fa': 65, 'Ex': 12, 'Po': 3})
# print(Counter(x_merged['ExterCond']))
# CHANGE IT into numerical: increasing values means better
x_merged['ExterCond'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
numeric_feat.append('ExterCond')

# ADD NEW FEATURE: ExterQualCondMean
x_merged['ExterQualCondMean'] = round(x_merged[['ExterQual', 'ExterCond']].mean(axis=1))
numeric_feat.append('ExterQualCondMean')
new_feat.append('ExterQualCondMean')

# ADD NEW FEATURE: ExterQualCond
# x_merged['ExterQualCond'] = x_merged['ExterQual'] + x_merged['ExterCond']
# numeric_feat.append('ExterQualCond')
# new_feat.append('ExterQualCond')

# Foundation: Type of foundation
# 0/2909 null values
# ({'PConc': 1304, 'CBlock': 1232, 'BrkTil': 310, 'Slab': 49, 'Stone': 11, 'Wood': 5})
# print(Counter(x_merged['Foundation']))
one_hot_enc_feat.append('Foundation')

# ADD NEW FEATURE: FoundationQual that is a numeric version of Foundation (increasing values means better)
x_merged['FoundationQual'] = x_merged['Foundation'].replace({'Wood': 1, 'Stone': 2, 'Slab': 3, 'PConc': 4,
                                                             'CBlock': 5, 'BrkTil': 6})
numeric_feat.append('FoundationQual')
new_feat.append('FoundationQual')

# BsmtQual: Evaluates the height of the basement
# 81/2909 null values
# 79/2909 NO BASEMENT
# ({'TA': 1280, 'Gd': 1206, 'Ex': 256, 'Fa': 88, nan: 81})
# print(Counter(x_merged['BsmtQual']))
# to understand how to fill nan value
bsmt = x_merged[['BsmtQual', 'BsmtCond']]
bsmt_null = (bsmt[(bsmt['BsmtCond'].isnull()) | (bsmt['BsmtQual'].isnull())])
# REPLACE: 2209, 2210 (have a value for BsmtCond), others with 0
x_merged.loc[2209, 'BsmtQual'] = 'TA'
x_merged.loc[2210, 'BsmtQual'] = 'TA'
x_merged['BsmtQual'] = x_merged['BsmtQual'].fillna(0)
# CHANGE IT into numerical: increasing values means better
x_merged['BsmtQual'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
numeric_feat.append('BsmtQual')

# BsmtCond: Evaluates the general condition of the basement
# 82/2909 null values
# 79/2909 NO BASEMENT
# ({'TA': 2600, 'Gd': 121, 'Fa': 103, nan: 82, 'Po': 5})
# print(Counter(x_merged['BsmtCond']))
# REPLACE: 2032, 2177, 2516 (have a value for BsmtQual), others with 0
x_merged.loc[2032, 'BsmtCond'] = 'TA'
x_merged.loc[2177, 'BsmtCond'] = 'TA'
x_merged.loc[2516, 'BsmtCond'] = 'TA'
x_merged['BsmtCond'] = x_merged['BsmtCond'].fillna(0)
# CHANGE IT into numerical: increasing values means better
x_merged['BsmtCond'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
numeric_feat.append('BsmtCond')

# ADD NEW FEATURE: mean between ExterQual and BsmtCond
x_merged['BsmtQualCondMean'] = round(x_merged[['BsmtQual', 'BsmtCond']].mean(axis=1))
numeric_feat.append('BsmtQualCondMean')
new_feat.append('BsmtQualCondMean')

# BsmtExposure: Refers to walkout or garden level walls
# 82/2909 null values
# 79/2909 NO BASEMENT
# ({'No': 1900, 'Av': 418, 'Gd': 273, 'Mn': 238, nan: 82})
# print(Counter(x_merged['BsmtExposure']))
# to understand how to fill nan value
bsmt = x_merged[['BsmtQual', 'BsmtCond', 'BsmtExposure']]
bsmt_exp_null = (bsmt[bsmt['BsmtExposure'].isnull()])
# REPLACE: 942, 1479, 2340 (have a value for BsmtCond and BsmtQual), others with 0
x_merged.loc[942, 'BsmtExposure'] = 'No'
x_merged.loc[1479, 'BsmtExposure'] = 'No'
x_merged.loc[2340, 'BsmtExposure'] = 'No'
x_merged['BsmtExposure'] = x_merged['BsmtExposure'].fillna(0)
# CHANGE IT into numerical: increasing values means better
# NO IS DIFFERENT FROM NA!!
x_merged['BsmtExposure'].replace({'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, inplace=True)
numeric_feat.append('BsmtExposure')

# BsmtFinType1: Rating of basement finished area
# 79/2909 null values -> NO BASEMENT
# ({'Unf': 849, 'GLQ': 845, 'ALQ': 429, 'Rec': 287, 'BLQ': 268, 'LwQ': 154, nan: 79})
# print(Counter(x_merged['BsmtFinType1']))
# REPLACE: fillna with 0
# CHANGE IT into numerical: increasing values means better
x_merged['BsmtFinType1'] = x_merged['BsmtFinType1'].fillna(0)
x_merged['BsmtFinType1'].replace({'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, inplace=True)
numeric_feat.append('BsmtFinType1')

# BsmtFinSF1: Type 1 finished square feet
x_merged['BsmtFinSF1'] = x_merged['BsmtFinSF1'].fillna(0)
numeric_feat.append('BsmtFinSF1')

# BsmtFinType2: Rating of basement finished area (if multiple types)
# 79/2909 null values -> NO BASEMENT
# ({'Unf': 2486, 'Rec': 105, 'LwQ': 87, nan: 80, 'BLQ': 67, 'ALQ': 52, 'GLQ': 34})
# print(Counter(x_merged['BsmtFinType2']))
# to understand how to fill nan value
bsmt = x_merged[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinType2']]
bsmt_type2_null = (bsmt[bsmt['BsmtFinType2'].isnull()])
# REPLACE: 330 (have a value for other 4 features, so it has a basement), others with 0
x_merged.loc[330, 'BsmtFinType2'] = 'Unf'
x_merged['BsmtFinType2'] = x_merged['BsmtFinType2'].fillna(0)
# CHANGE IT into numerical: increasing values means better
x_merged['BsmtFinType2'] = x_merged['BsmtFinType2'].fillna(0)
x_merged['BsmtFinType2'].replace({'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, inplace=True)
numeric_feat.append('BsmtFinType2')

# ADD NEW FEATURE: HasMultipleBsmtType 0,1
x_merged['HasMultipleBsmtType'] = np.where(x_merged['BsmtFinType1'] != x_merged['BsmtFinType2'], 1, 0)
new_feat.append('HasMultipleBsmtType')

# BsmtFinSF2: Type 2 finished square feet
x_merged['BsmtFinSF2'] = x_merged['BsmtFinSF2'].fillna(0)
numeric_feat.append('BsmtFinSF2')

# BsmtUnfSF: Unfinished square feet of basement area
x_merged['BsmtUnfSF'] = x_merged['BsmtUnfSF'].fillna(0)
numeric_feat.append('BsmtUnfSF')

# TotalBsmtSF: Total square feet of basement area
x_merged['TotalBsmtSF'] = x_merged['TotalBsmtSF'].fillna(0)
numeric_feat.append('TotalBsmtSF')

# Heating: Type of heating
# 0/2909 null values
# ({'GasA': 2866, 'GasW': 27, 'Grav': 9, 'Wall': 6, 'OthW': 2, 'Floor': 1})
# print(Counter(x_merged['Heating']))
heating_check = x_merged['Heating'][(x_merged['Heating'] == 'OthW') | (x_merged['Heating'] == 'Floor')]
one_hot_enc_feat.append('Heating')

# HeatingQC: Heating quality and condition
# 0/2909 null values
# ({'Ex': 1488, 'TA': 855, 'Gd': 473, 'Fa': 92, 'Po': 3})
# print(Counter(x_merged['HeatingQC']))
# CHANGE IT into numerical: increasing values means better
x_merged['HeatingQC'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
numeric_feat.append('HeatingQC')

# CentralAir: Central air conditioning
# 0/2909 null values
# ({'Y': 2717, 'N': 194})
# print(Counter(x_merged['CentralAir']))
# CHANGE IT into boolean
x_merged['CentralAir'].replace({'Y': 1, 'N': 0}, inplace=True)

# Electrical: Electrical system
# 0/2909 null values
# ({'SBrkr': 2663, 'FuseA': 188, 'FuseF': 50, 'FuseP': 8, 'Mix': 1, nan: 1})
# print(Counter(x_merged['Electrical']))
# REPLACE: with mode SBrkr
x_merged['Electrical'] = x_merged['Electrical'].fillna('SBrkr')
electr_check = x_merged['Electrical'][(x_merged['Electrical'] == 'Mix')]
one_hot_enc_feat.append('Electrical')

# ADD NEW FEATURE: HasBsmt
x_merged['HasBsmt'] = np.where(x_merged['BsmtQual'] == 0, 0, 1)
new_feat.append('HasBsmt')

# 1stFlrSF: First Floor square feet
# 0/2909 null values
# print('1stFlrSF: ', x_merged['1stFlrSF'].isnull().sum())
numeric_feat.append('1stFlrSF')

# 2ndFlrSF: Second floor square feet
# 0/2909 null values
# print('2ndFlrSF: ', x_merged['2ndFlrSF'].isnull().sum())
numeric_feat.append('2ndFlrSF')

# ADD NEW FEATURE: Has2ndFlr
x_merged['Has2ndFlr'] = np.where(x_merged['2ndFlrSF'] > 0, 1, 0)
new_feat.append('Has2ndFlr')

# ADD NEW FEATURE: Total floor square feet
x_merged['TotalSF'] = x_merged['TotalBsmtSF'] + x_merged['1stFlrSF'] + x_merged['2ndFlrSF']
new_feat.append('TotalSF')

# LowQualFinSF: Low quality finished square feet (all floors)
# 0/2909 null values
# print('LowQualFinSF: ', x_merged['LowQualFinSF'].isnull().sum())
numeric_feat.append('LowQualFinSF')

# GrLivArea: Above grade (ground) living area square feet
# 0/2909 null values
# print('GrLivArea: ', x_merged['GrLivArea'].isnull().sum())
numeric_feat.append('GrLivArea')

# BsmtFullBath: Basement full bathrooms
# 2/2909 null values
# print('BsmtFullBath: ', x_merged['BsmtFullBath'].isnull().sum())
# REPLACE: with median
x_merged['BsmtFullBath'] = x_merged['BsmtFullBath'].transform(lambda x: x.fillna(x.median()))
numeric_feat.append('BsmtFullBath')

# BsmtHalfBath: Basement half bathrooms
# 2/2909 null values
# print('BsmtHalfBath: ', x_merged['BsmtHalfBath'].isnull().sum())
# REPLACE: with median
x_merged['BsmtHalfBath'] = x_merged['BsmtHalfBath'].transform(lambda x: x.fillna(x.median()))
numeric_feat.append('BsmtHalfBath')

# FullBath: Full bathrooms above grade
# 0/2909 null values
# print('FullBath: ', x_merged['FullBath'].isnull().sum())
numeric_feat.append('FullBath')

# HalfBath: Half baths above grade
# 0/2909 null values
# print('HalfBath: ', x_merged['HalfBath'].isnull().sum())
numeric_feat.append('HalfBath')

# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
# THE REAL NAME IS BedroomAbvGr !!!
# 0/2909 null values
# print('BedroomAbvGr: ', x_merged['BedroomAbvGr'].isnull().sum())
numeric_feat.append('BedroomAbvGr')

# Kitchen: Kitchens above grade
# THE REAL NAME IS KitchenAbvGr !!!
# 0/2909 null values
# print('KitchenAbvGr: ', x_merged['KitchenAbvGr'].isnull().sum())
numeric_feat.append('KitchenAbvGr')

# KitchenQual: Kitchen quality
# 1/2909 null values
# Counter({'TA': 1490, 'Gd': 1148, 'Ex': 203, 'Fa': 69, nan: 1})
# print(Counter(x_merged['KitchenQual']))
# REPLACE: with most common (TA)
x_merged['KitchenQual'] = x_merged['KitchenQual'].fillna('TA')
# CHANGE IT into numerical: increasing values means better
x_merged['KitchenQual'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
numeric_feat.append('KitchenQual')

# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# print('TotRmsAbvGrd: ', x_merged['TotRmsAbvGrd'].isnull().sum())
numeric_feat.append('KitchenAbvGr')

# Functional: Home functionality (Assume typical unless deductions are warranted)
# 2/2909 null values
# ({'Typ': 2710, 'Min2': 70, 'Min1': 64, 'Mod': 35, 'Maj1': 19, 'Maj2': 9, 'Sev': 2, nan: 2})
# print(Counter(x_merged['Functional']))
# REPLACE: assume typical unless deductions are warranted
x_merged['Functional'] = x_merged['Functional'].fillna('Typ')
# check if Sev and Maj2 are also in the test set
fun_not_typ = x_merged['Functional'][(x_merged['Functional'] == 'Sev') | (x_merged['Functional'] == 'Maj2')]
one_hot_enc_feat.append('Functional')

# ADD NEW FEATURE: FunctionalQual that is a numeric version of Functional (increasing values means better)
x_merged['FunctionalQual'] = x_merged['Functional'].replace({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5,
                                                             'Min2': 6, 'Min1': 7, 'Typ': 8})
numeric_feat.append('FunctionalQual')
new_feat.append('FunctionalQual')

# Fireplaces: Number of fireplaces
# 0/2909 null values
# 1418/2909 no fireplaces
# print('fireplaces: ', x_merged['Fireplaces'].isnull().sum())
no_fireplaces = (x_merged['Fireplaces'][x_merged['Fireplaces'] == 0]).count()
numeric_feat.append('Fireplaces')

# FireplaceQu: Fireplace quality
# 1418/2909 null values -> NO FIREPLACES
# ({nan: 1418, 'Gd': 740, 'TA': 591, 'Fa': 74, 'Po': 45, 'Ex': 43})
# print(Counter(x_merged['FireplaceQu']))
# REPLACE: nan means NO FIREPLACES
x_merged['FireplaceQu'] = x_merged['FireplaceQu'].fillna(0)
# CHANGE IT into numerical: increasing values means better
x_merged['FireplaceQu'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
numeric_feat.append('FireplaceQu')

# ADD NEW FEATURE: HasFireplaces 0,1
x_merged['HasFireplaces'] = np.where(x_merged['FireplaceQu'] == 0, 0, 1)
new_feat.append('HasFireplaces')

# GarageType: Garage location
# 156/2909 null values -> NO GARAGE
# 157/2909 after adjust
# ({'Attchd': 1719, 'Detchd': 777, 'BuiltIn': 185, nan: 156, 'Basment': 36, '2Types': 23, 'CarPort': 15})
# print(Counter(x_merged['GarageType']))
# adjust value record 2568, 2118
x_merged.loc[2568, 'GarageType'] = 'NoGarage'
x_merged.loc[2118, 'GarageType'] = 'NoGarage'
# REPLACE: nan means NO GARAGE
x_merged['GarageType'] = x_merged['GarageType'].fillna('NoGarage')
one_hot_enc_feat.append('GarageType')

# GarageYrBlt: Year garage was built
# 158/2909 null values (different from 156..)
# print('GarageYrBlt: ', x_merged['GarageYrBlt'].isnull().sum())
x_merged['GarageYrBlt'] = x_merged['GarageYrBlt'].fillna(0)
no_gar = (x_merged['GarageYrBlt'][x_merged['GarageYrBlt'] == 0]).count()
numeric_feat.append('GarageYrBlt')

# GarageFinish: Interior finish of the garage
# 158/2909 null values -> NO GARAGE (different from 156..)
# ({'Unf': 1228, 'RFn': 809, 'Fin': 716, nan: 158})
# print(Counter(x_merged['GarageFinish']))
# REPLACE: nan means NO GARAGE
x_merged['GarageFinish'] = x_merged['GarageFinish'].fillna(0)
one_hot_enc_feat.append('GarageFinish')

# GarageCars: Size of garage in car capacity
# 1/2909 null values
# 156/2909 NO GARAGE CARS
# 157/2909 after fillna
# print('GarageCars: ', x_merged['GarageCars'].isnull().sum())
# adjust value record 2118
x_merged.loc[2118, 'GarageCars'] = 0
# REPLACE: nan means NO GARAGE
x_merged['GarageCars'] = x_merged['GarageCars'].fillna(0)
no_gar_car = (x_merged['GarageCars'][x_merged['GarageCars'] == 0]).count()
numeric_feat.append('GarageCars')

# GarageArea: Size of garage in square feet
# 1/2909 null values
# 156/2909 NO GARAGE AREA
# 157/2909 after fillna
# print('GarageArea: ', x_merged['GarageArea'].isnull().sum())
# adjust value record 2118
x_merged.loc[2118, 'GarageArea'] = 0
# REPLACE: nan means NO GARAGE
x_merged['GarageArea'] = x_merged['GarageArea'].fillna(0)
no_gar_area = (x_merged['GarageArea'][x_merged['GarageArea'] == 0]).count()
numeric_feat.append('GarageArea')

# GarageQual: Garage quality
# 158/2909 null values -> NO GARAGE (different from 156..)
# ({'TA': 2597, nan: 158, 'Fa': 124, 'Gd': 24, 'Po': 5, 'Ex': 3})
# print(Counter(x_merged['GarageQual']))
# REPLACE: nan means NO GARAGE
x_merged['GarageQual'] = x_merged['GarageQual'].fillna(0)
# CHANGE IT into numerical: increasing values means better
x_merged['GarageQual'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
numeric_feat.append('GarageQual')

# GarageCond: Garage condition
# 158/2909 null values -> NO GARAGE (different from 156..)
# ({'TA': 2648, nan: 158, 'Fa': 73, 'Gd': 15, 'Po': 14, 'Ex': 3})
# print(Counter(x_merged['GarageCond']))
# REPLACE: nan means NO GARAGE
x_merged['GarageCond'] = x_merged['GarageCond'].fillna(0)
# CHANGE IT into numerical: increasing values means better
x_merged['GarageCond'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
numeric_feat.append('GarageCond')

# try to understand 156 vs 158 OK!
garage = x_merged[['GarageYrBlt', 'GarageArea', 'GarageCars', 'GarageFinish', 'GarageType', 'GarageQual', 'GarageCond']]
no_garage = garage[(x_merged['GarageCond'] == 0)]

# ADD NEW FEATURE GarageQualCondMean
x_merged['GarageQualCondMean'] = round(x_merged[['GarageQual', 'GarageCond']].mean(axis=1))
new_feat.append('GarageQualCondMean')
numeric_feat.append('GarageQualCondMean')

# ADD NEW FEATURE: HasFireplaces 0,1
x_merged['HasGarage'] = np.where(x_merged['GarageYrBlt'] == 0, 0, 1)
new_feat.append('HasGarage')

# PavedDrive: Paved driveway
# 0/2909 null values
# ({'Y': 2635, 'N': 214, 'P': 62})
# print(Counter(x_merged['PavedDrive']))
feat_to_drop.append('PavedDrive')

# ADD NEW FEATURE: HasPavedDrive 0,1
x_merged['HasPavedDrive'] = np.where(x_merged['PavedDrive'] == 'Y', 1, 0)
new_feat.append('HasPavedDrive')

# WoodDeckSF: Wood deck area in square feet
# 0/2909 null values
# 1520/2909 NO WOOD DECK
# print('WoodDeckSF: ', x_merged['WoodDeckSF'].isnull().sum())
no_wooDeck = (x_merged['WoodDeckSF'][(x_merged['WoodDeckSF'] == 0)]).count()

# ADD NEW FEATURE: HasWoodDeck 0,1
x_merged['HasWoodDeck'] = np.where(x_merged['WoodDeckSF'] == 0, 0, 1)
new_feat.append('HasWoodDeck')

# OpenPorchSF: Open porch area in square feet
# 0/2909 null values
# 1296/2909 NO OPEN PORCH
# print('OpenPorchSF: ', x_merged['OpenPorchSF'].isnull().sum())
no_openPorch = (x_merged['OpenPorchSF'][(x_merged['OpenPorchSF'] == 0)]).count()

# ADD NEW FEATURE: HasOpenPorch 0,1
x_merged['HasOpenPorch'] = np.where(x_merged['OpenPorchSF'] == 0, 0, 1)
new_feat.append('HasOpenPorch')

# EnclosedPorch: Enclosed porch area in square feet
# 0/2909 null values
# 2455/2909 NO ENCLOSED PORCH
# print('EnclosedPorch: ', x_merged['EnclosedPorch'].isnull().sum())
no_enclPorch = (x_merged['EnclosedPorch'][(x_merged['EnclosedPorch'] == 0)]).count()

# ADD NEW FEATURE: HasEnclPorch 0,1
x_merged['HasEnclPorch'] = np.where(x_merged['EnclosedPorch'] == 0, 0, 1)
new_feat.append('HasEnclPorch')

# 3SsnPorch: Three season porch area in square feet
# 0/2909 null values
# 2874/2909 NO 4 SEASON PORCH
# print('3SsnPorch: ', x_merged['3SsnPorch'].isnull().sum())
no_3seasPorch = (x_merged['3SsnPorch'][(x_merged['3SsnPorch'] == 0)]).count()
numeric_feat.append('3SsnPorch')
# feat_to_drop.append('3SsnPorch')

# ADD NEW FEATURE: Has3SsnPorch 0,1
x_merged['Has3SsnPorch'] = np.where(x_merged['3SsnPorch'] == 0, 0, 1)
new_feat.append('Has3SsnPorch')

# ScreenPorch: Screen porch area in square feet
# 0/2909 null values
# 2655/2909 NO SCREEN PORCH
# print('ScreenPorch: ', x_merged['ScreenPorch'].isnull().sum())
no_screenPorch = (x_merged['ScreenPorch'][(x_merged['ScreenPorch'] == 0)]).count()
numeric_feat.append('ScreenPorch')
# feat_to_drop.append('ScreenPorch')

# ADD NEW FEATURE: HasScreenPorch 0,1
x_merged['HasScreenPorch'] = np.where(x_merged['ScreenPorch'] == 0, 0, 1)
new_feat.append('HasScreenPorch')

# PoolArea: Pool area in square feet
# 0/2909 null values
# 2899/2909 NO POOL
# print('PoolArea: ', x_merged['PoolArea'].isnull().sum())
no_pool = (x_merged['PoolArea'][(x_merged['PoolArea'] == 0)]).count()
numeric_feat.append('PoolArea')
# feat_to_drop.append('PoolArea')

# ADD NEW FEATURE: HasPool 0,1
x_merged['HasPool'] = np.where(x_merged['PoolArea'] == 0, 0, 1)
new_feat.append('HasPool')

# PoolQC: Pool quality
# 2902/2909 null values
# ({nan: 2902, 'Ex': 4, 'Gd': 3, 'Fa': 2})
# print(Counter(x_merged['PoolQC']))
feat_to_drop.append('PoolQC')

# Fence: Fence quality
# 2902/2909 null values
# ({nan: 2343, 'MnPrv': 327, 'GdPrv': 118, 'GdWo': 111, 'MnWw': 12})
# print(Counter(x_merged['Fence']))
# REPLACE: nan means NO FENCE
x_merged['Fence'] = x_merged['Fence'].fillna(0)
# CHANGE IT into numerical: increasing values means better
x_merged['Fence'].replace({'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}, inplace=True)
numeric_feat.append('Fence')

# MiscFeature: Miscellaneous feature not covered in other categories
# 2902/2909 null values
# ({nan: 2806, 'Shed': 95, 'Gar2': 5, 'Othr': 4, 'TenC': 1})
# print(Counter(x_merged['MiscFeature']))
# REPLACE: nan means NO ADD FEATURES
x_merged['MiscFeature'] = x_merged['MiscFeature'].fillna('NoFeatures')
one_hot_enc_feat.append('MiscFeature')

# MiscVal: $Value of miscellaneous feature
# 0/2909 null values
# 2808/2909 NO MISC VAL
# print('MiscVal: ', x_merged['MiscVal'].isnull().sum())
no_misc_val = (x_merged['MiscVal'][(x_merged['MiscVal'] == 0)]).count()

# try to understand misc value/feat missing values
misc = x_merged[['MiscFeature', 'MiscVal']]
no_misc = misc[(misc['MiscVal'] == 0) & (misc['MiscFeature'] != 'NoFeatures')]

# adjust values (shed_mean = 764, othr_mean = 3250)
x_merged.loc[867, 'MiscVal'] = 3250
x_merged.loc[1194, 'MiscVal'] = 764
x_merged.loc[2423, 'MiscVal'] = 764

# MoSold: Month Sold (MM)
# 0/2909 null values
# ({6: 502, 7: 445, 5: 394, 4: 279, 8: 232, 3: 231, 10: 171, 9: 158, 11: 142, 2: 133, 1: 121, 12: 103})
# print(Counter(x_merged['MoSold']))
# CHANGE IT into categorical
x_merged['MoSold'].replace({1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"})

# YrSold: Year Sold (YYYY)
# 0/2909 null values
# ({2007: 690, 2009: 645, 2008: 619, 2006: 619, 2010: 338})
# print(Counter(x_merged['YrSold']))
numeric_feat.append('YrSold')

# ADD NEW FEATURE YearsSinceRemodel
x_merged['YearsSinceRemodel'] = x_merged['YrSold'].astype(int) - x_merged['YearRemodAdd'].astype(int)

# SaleType: Type of sale
# 1/2909 null values
# ({'WD': 2520, 'New': 237, 'COD': 87, 'ConLD': 25, 'CWD': 12, 'ConLI': 9, 'ConLw': 8, 'Oth': 7, 'Con': 5, nan: 1})
# print(Counter(x_merged['SaleType']))
# REPLACE: nan with mode (WD)
x_merged['SaleType'] = x_merged['SaleType'].fillna('WD')
# CHANGE IT into numerical: increasing values means better
x_merged['SaleType'].replace({'Oth': 0, 'ConLD': 1, 'ConLI': 2, 'ConLw': 3, 'Con': 4, 'COD': 5, 'New': 6, 'VWD': 7,
                              'CWD': 8, 'WD': 9}, inplace=True)
numeric_feat.append('SaleType')

# SaleCondition: Condition of sale
# 0/2909 null values
# ({'Normal': 2397, 'Partial': 243, 'Abnorml': 189, 'Family': 46, 'Alloca': 24, 'AdjLand': 12})
# print(Counter(x_merged['SaleCondition']))
one_hot_enc_feat.append('SaleCondition')

# feat to drop: ['LotShape', 'Utilities', 'PoolQC']
x_merged.drop(feat_to_drop, inplace=True, axis=1)

# check missing values -> NO MISSING VALUE!
missing_value = UtilFunct.check_missing_data(x_merged)

# get dummies & drop columns to avoid overfit
x_merged = pd.get_dummies(x_merged, columns=one_hot_enc_feat)
overfit_cols = ['Electrical_Mix', 'Heating_Floor', 'Heating_OthW',
                'RoofMatl_Metal', 'RoofMatl_Membran', 'RoofMatl_Roll']
x_merged.drop(overfit_cols, inplace=True, axis=1)

# separate train and test
x_train = x_merged[:1452]
x_test = x_merged[1452:]

# REGRESSION
RANDOM_STATE = 42
kfold = KFold(n_splits=20, shuffle=False, random_state=RANDOM_STATE)

scaler = RobustScaler()
X_train = scaler.fit_transform(x_train)
Y_train = y_train
X_test = scaler.transform(x_test)

stacked_pred = Models.stacked_regression(X_train, Y_train, X_test, RANDOM_STATE, kfold)
xgb_pred = Models.xgb_regression(X_train, Y_train, X_test, RANDOM_STATE, kfold)

final_pred = (0.77*stacked_pred+0.23*xgb_pred)

# create submission file
UtilFunct.create_submission_file(final_pred)