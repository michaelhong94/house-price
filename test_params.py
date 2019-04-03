import sys
sys.path.append('./xgboost/python-package/')
sys.path.append('./fancyimpute/fancyimpute/')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from mlxtend.regressor import StackingRegressor
pd.set_option("display.max_columns",1000000)
pd.set_option('display.max_rows', 1000000)
from fancyimpute import KNN
train = pd.read_csv('./remastered.csv')
test = pd.read_csv('./test_prepro.csv')
train_target = train['SalePrice']
train = train.drop(['index','Id'],axis = 1)
test = test.drop(['index','Id'],axis = 1)
test[['GarageYrBlt','YrSold']]
train['MSSubClass'] = train['MSSubClass'].astype(object)
test['MSSubClass'] = test['MSSubClass'].astype(object)
lis = ['YrSold','YearBuilt','YearRemodAdd','GarageYrBlt']
train[lis] = train[lis].astype(int)
test[lis] = test[lis].astype(int)
train_without4_without_price = train.drop(['LotFrontage','MasVnrArea','MasVnrType','Electrical','SalePrice'],axis=1)
test_without4 = test.drop(['LotFrontage','MasVnrArea','MasVnrType','Electrical'],axis=1)


def transform(X):
    X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
    X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
    X['Yrsold_minus_Remod'] = X['YrSold'] - X['YearRemodAdd']
    # X[X['Yrsold_minus_Remod'] < 0] = 0.9
    X.loc[X['Yrsold_minus_Remod'] < 0, 'Yrsold_minus_Remod'] = 0.9
    X['Yrsold_minus_garage'] = X['YrSold'] - X['GarageYrBlt']
    X.loc[X['Yrsold_minus_garage'] < 0, 'Yrsold_minus_garage'] = 0.9
    X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
    X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]

    X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]

    X["-_Functional_TotalHouse"] = X["Functional"] * X["TotalHouse"]
    X["-_Functional_OverallQual"] = X["Functional"] + X["OverallQual"]
    X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
    X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]

    X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
    X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]
    X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
    X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"] + X[
        "EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]

    return X


X_numeric=train_without4_without_price.select_dtypes(exclude=["object"])
skewness = X_numeric.apply(lambda x: skew(x))
skewness_features = skewness[abs(skewness) >= 0.75].index
train_without4_without_price[skewness_features] = np.log1p(train_without4_without_price[skewness_features])
train_without4_without_price = pd.get_dummies(train_without4_without_price)

X_numeric=test_without4.select_dtypes(exclude=["object"])
# skewness = X_numeric.apply(lambda x: skew(x))
# skewness_features = skewness[abs(skewness) >= 0.75].index
test_without4[skewness_features] = np.log1p(test_without4[skewness_features])
test_without4 = pd.get_dummies(test_without4)

full = pd.concat([train_without4_without_price,test_without4])

full = full.fillna(0)

train_without4_without_price_want_to_impute = full.iloc[:1460,:]
test_without4_want_to_impute = full.iloc[1460:,:]

train_only4 = train[['LotFrontage','MasVnrArea','MasVnrType','Electrical']]
test_only4 = test[['LotFrontage','MasVnrArea','MasVnrType','Electrical']]

train_with4 = pd.concat([train_without4_without_price_want_to_impute,train_only4],axis=1)
test_with4 = pd.concat([test_without4_want_to_impute,test_only4],axis=1)

all_column_name = list(train_with4.columns)
X_filled_knn_train = KNN().fit_transform(train_with4)
X_filled_knn_test = KNN().fit_transform(test_with4)
fullfill_train = pd.DataFrame(X_filled_knn_train,columns=all_column_name)
fullfill_test = pd.DataFrame(X_filled_knn_test,columns=all_column_name)

fullfill_train['SalePrice'] = train_target

fullfill_train_X = fullfill_train.drop(['SalePrice'],axis = 1)

train_without4_without_price = train.drop(['LotFrontage','MasVnrArea','MasVnrType','Electrical','SalePrice'],axis=1)
test_without4 = test.drop(['LotFrontage','MasVnrArea','MasVnrType','Electrical'],axis=1)

train_with4 = pd.concat([train_without4_without_price,fullfill_train[['LotFrontage','MasVnrArea','MasVnrType','Electrical']]],axis = 1)

test_with4 = pd.concat([test_without4,fullfill_test[['LotFrontage','MasVnrArea','MasVnrType','Electrical']]],axis = 1)

train_with4 = transform(train_with4)

test_with4 = transform(test_with4)

train_with4.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold','MoSold'],axis = 1,inplace=True)

test_with4.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold','MoSold'],axis = 1,inplace=True)

full2 = pd.concat([train_with4,test_with4])

full2 = full2.fillna(0)

train_with4 = full2.iloc[:1460,:]
test_with4 = full2.iloc[1460:,:]

X_numeric=train_with4.select_dtypes(exclude=["object"])
skewness = X_numeric.apply(lambda x: skew(x))
skewness_features = skewness[abs(skewness) >= 0.75].index
train_with4[skewness_features] = np.log1p(train_with4[skewness_features])
train_with4 = pd.get_dummies(train_with4)

X_numeric=test_with4.select_dtypes(exclude=["object"])
# skewness = X_numeric.apply(lambda x: skew(x))
# skewness_features = skewness[abs(skewness) >= 0.75].index
test_with4[skewness_features] = np.log1p(test_with4[skewness_features])
test_with4 = pd.get_dummies(test_with4)

full3 = pd.concat([train_with4,test_with4])
full3 = full3.fillna(0)
train_with4 = full3.iloc[:1460,:]
test_with4 = full3.iloc[1460:,:]

scaler = RobustScaler()
X_scaled_train = scaler.fit(train_with4).transform(train_with4)
# X_scaled_test = scaler.fit(test_with4).transform(test_with4)

X_scaled_test = scaler.fit(train_with4).transform(test_with4)

log_train_target = np.log(train_target)

lasso=Lasso(alpha=0.001)
lasso.fit(X_scaled_train,log_train_target)
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=train_with4.columns)
FI_lasso.sort_values("Feature Importance",ascending=False)


FI_lasso.sort_values("Feature Importance",ascending=False).index[:50]

FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()

train_with4 = train_with4[['OverallQual', 'Neighborhood_StoneBr', '-_Functional_TotalHouse',
       'Neighborhood_NridgHt', 'Neighborhood_NoRidge', 'Neighborhood_Crawfor',
       'GrLivArea', 'Condition1_Norm', 'Neighborhood_Somerst',
       'Exterior1st_BrkFace', 'GarageCars', 'BsmtFinSF1',
       '-_TotalHouse_LotArea', 'OverallCond', 'HeatingQC', 'KitchenQual',
       'FullBath', 'HalfBath', 'TotalPlace', 'TotalHouse', 'BsmtQual',
       'BsmtFullBath', 'FireplaceQu', 'SaleType_New', 'RoofMatl',
       'Neighborhood_BrkSide', 'BsmtExposure', 'BsmtFinType1', 'BldgType_1Fam',
       'Fireplaces', 'WoodDeckSF', 'Exterior2nd_VinylSd', 'ExterQual',
       'Exterior1st_MetalSd', '1stFlrSF', 'GarageFinish', 'RoofStyle_Hip',
       'ScreenPorch', 'SaleCondition_Normal', 'TotalArea', 'MSZoning_RL',
       'LowQualFinSF', 'Rooms', 'OpenPorchSF', '3SsnPorch', 'PavedDrive',
       'CentralAir_Y', 'MiscFeature_TenC', 'Neighborhood_ClearCr',
       'LandContour']]

test_with4 = test_with4[['OverallQual', 'Neighborhood_StoneBr', '-_Functional_TotalHouse',
       'Neighborhood_NridgHt', 'Neighborhood_NoRidge', 'Neighborhood_Crawfor',
       'GrLivArea', 'Condition1_Norm', 'Neighborhood_Somerst',
       'Exterior1st_BrkFace', 'GarageCars', 'BsmtFinSF1',
       '-_TotalHouse_LotArea', 'OverallCond', 'HeatingQC', 'KitchenQual',
       'FullBath', 'HalfBath', 'TotalPlace', 'TotalHouse', 'BsmtQual',
       'BsmtFullBath', 'FireplaceQu', 'SaleType_New', 'RoofMatl',
       'Neighborhood_BrkSide', 'BsmtExposure', 'BsmtFinType1', 'BldgType_1Fam',
       'Fireplaces', 'WoodDeckSF', 'Exterior2nd_VinylSd', 'ExterQual',
       'Exterior1st_MetalSd', '1stFlrSF', 'GarageFinish', 'RoofStyle_Hip',
       'ScreenPorch', 'SaleCondition_Normal', 'TotalArea', 'MSZoning_RL',
       'LowQualFinSF', 'Rooms', 'OpenPorchSF', '3SsnPorch', 'PavedDrive',
       'CentralAir_Y', 'MiscFeature_TenC', 'Neighborhood_ClearCr',
       'LandContour']]

scaler = RobustScaler()
X_scaled_train = scaler.fit(train_with4).transform(train_with4)
X_scaled_test = scaler.fit(train_with4).transform(test_with4)

pca = PCA().fit(X_scaled_train)

position = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.99)[0][0]

cumsum = np.cumsum(pca.explained_variance_)
fig, ax1 = plt.subplots()
ax1.bar(range(0,50),pca.explained_variance_ratio_,label = 'explained_variance_ratio_')
ax1.set_xlabel('components')
ax1.set_ylabel('explained_variance_ratio_')

ax2 = ax1.twinx()
ax2.plot(range(0,50),cumsum,label='cumsum',color='g')
position = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.99)[0][0]
ax1.vlines(position,0,0.08,label='where 99% variance is')
ax1.legend(loc = 'right')
ax2.legend(loc = 'lower right')
plt.show()


pca = PCA(n_components=37)
train_after_pca_87 = pca.fit_transform(X_scaled_train)
test_after_pca_87 = pca.transform(X_scaled_test)

lr = LinearRegression()
lr.fit(train_after_pca_87,log_train_target)


def alpha_calculation(price_origin_train, X_post_PCA, imputed_data, listype, model):
    # assign values to X,y:
    data_X = imputed_data.drop('SalePrice', axis=1)
    data_y = imputed_data['SalePrice']
    log_y = np.log(data_y)

    # PCA transformation:
    # redo_data_X = scaler.fit(data_X).transform(data_X)
    # trans_redo_data_X = pca.fit_transform(redo_data_X)

    # find index
    index = list(price_origin_train[price_origin_train['SaleCondition'] == listype].index)
    # print(index)

    # find the relevant values corresponding to the index
    trans_redo_data_X_dataframe = pd.DataFrame(X_post_PCA).ix[index]

    # print(remastered_train,train_X_after_PCA, train_data)

    # prediction
    pred = model.predict(trans_redo_data_X_dataframe.values)

    # find the alpha corresponding to the abnormal salescondition
    attribute_alpha = log_y.ix[index].values.T / pred

    alpha = attribute_alpha.mean()
    origin_mse = mean_squared_error(log_y.ix[index].values.T, pred.reshape(-1, 1))
    modified_mse = mean_squared_error(log_y.ix[index].values.T, alpha * pred.reshape(-1, 1))
    ratio = (origin_mse - modified_mse) / origin_mse
    return alpha, origin_mse, modified_mse, ratio, model



fa_alpha,fa_origin_mse,fa_modified_mse,fa_ratio,model = alpha_calculation(train,train_after_pca_87, fullfill_train,'Family',lr)

abn_alpha,abn_origin_mse,abn_modified_mse,abn_ratio,model = alpha_calculation(train,train_after_pca_87, fullfill_train,'Abnorml',lr)

adj_alpha,adj_origin_mse,adj_modified_mse,adj_ratio,model = alpha_calculation(train,train_after_pca_87, fullfill_train,'AdjLand',lr)

all_alpha,all_origin_mse,all_modified_mse,all_ratio,model = alpha_calculation(train,train_after_pca_87, fullfill_train,'Alloca',lr)

par_alpha,par_origin_mse,par_modified_mse,par_ratio,model = alpha_calculation(train,train_after_pca_87, fullfill_train,'Partial',lr)
print('fa_alpha: %f origin: %f modified: %f improve %f'%(fa_alpha,fa_origin_mse,fa_modified_mse,fa_ratio))

print('abn_alpha: %f origin: %f modified: %f improve %f'%(abn_alpha,abn_origin_mse,abn_modified_mse,abn_ratio))

print('adj_alpha: %f origin: %f modified: %f improve %f'%(adj_alpha,adj_origin_mse,adj_modified_mse,adj_ratio))

print('all_alpha: %f origin: %f modified: %f improve %f'%(all_alpha,all_origin_mse,all_modified_mse,all_ratio))

print('par_alpha: %f origin: %f modified: %f improve %f'%(par_alpha,par_origin_mse,par_modified_mse,par_ratio))

sale_condition = test.SaleCondition
alpha_dict = {'Normal':1,'Abnorml':abn_alpha,'AdjLand':adj_alpha,'Alloca':all_alpha,'Family':fa_alpha,'Partial':par_alpha}

pred_without_alpha_lr = lr.predict(test_after_pca_87)
pred_with_alpha_lr = pred_without_alpha_lr
for i in range(len(sale_condition)):
    pred_with_alpha_lr[i] = alpha_dict[sale_condition[i]] * pred_without_alpha_lr[i]
pred_with_alpha_lr = pd.DataFrame(data = np.exp(pred_with_alpha_lr),columns = ['SalePrice'],index = range(1461,2920))
pred_with_alpha_lr.index.name = 'Id'
pred_with_alpha_lr.to_csv('./pred_with_alpha_lr.csv',index = True)


def stacking_regressor_training(origin_data, data, log_y):
    '''
    description for input:
    {
    origin_data: 最原始数据；
    data： 用于模型训练的数据
    }
    description for output:
    {
    the best parameters:str
    }
    '''
    # Find SaleCondition=='Normal' observations to fit the hyperparameters
    normal_index = origin_data[origin_data['SaleCondition'] == 'Normal'].index
    data_to_choose_hyperparas_x = data.ix[normal_index]
    data_to_choose_hyperparas_y = log_y.ix[normal_index]

    # create the instance for the ensemble model
    xgboost = XGBRegressor(nthread=-1)
    """
    gbr = GradientBoostingRegressor(random_state=1,
                min_samples_split=2,
                max_features='sqrt',
                min_samples_leaf=1,)
    etr = ExtraTreesRegressor(bootstrap=False, criterion='mse', 
          min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
          min_weight_fraction_leaf=0.0,n_jobs=-1,
         random_state=0, verbose=0, warm_start=True)
    rfr = RandomForestRegressor()
    regressors = [xgboost, gbr, etr, rfr]
    stregr = StackingRegressor(regressors=regressors, meta_regressor=xgboost)
    """
    # fit the hyperparameters
    params = {
        'xgboost__learning_rate': [x / 20.0 for x in range(1, 20)],
        'xgboost__reg_alph': [x / 20.0 for x in range(1, 20)],
        'xgboost__reg_lambda': [x / 20.0 for x in range(1, 20)],
        'xgboost__n_estimators': [1000, 1200, 1500, 1800, 2000, 2200, 2400, 2600, 2800, 3000],
        'xgboost__subsample': [x / 20.0 for x in range(1, 20)],
        'xgboost__gamma': [x / 20.0 for x in range(1, 20)]
        #         """
        #         'rfr__n_estimators': [range(0,2100,100)]
        #         'gbr__n_estimators': [range(0,3100,100)]
        #         'gbr__learning_rate': [x/20.0 for range(1,20)]
        #         'gbr__subsample': [x/10.0 for range(1,10)]
        #         'gbr__max_depth': [range(3,6)]
        #         'etr__n_estimators': [range(0,3100,100)]
        #         'etr__learning_rate': [x/20.0 for range(1,20)]
        #         'etr__subsample': [x/10.0 for range(1,10)]
        #         'etr__max_depth': [range(3,6)]
        #         """

    }
    grid = GridSearchCV(estimator=xgboost, param_grid=params, cv=5, refit=True)
    grid.fit(data_to_choose_hyperparas_x, data_to_choose_hyperparas_y)

    return grid.best_params_

best_params = stacking_regressor_training(train,pd.DataFrame(train_after_pca_87),pd.DataFrame(log_train_target))