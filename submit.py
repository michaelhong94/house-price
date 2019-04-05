
# coding: utf-8

# # 首先，我们从递进编码之后的数据开始

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
#from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import GridSearchCV
#from mlxtend.regressor import StackingRegressor
pd.set_option("display.max_columns",1000000)
pd.set_option('display.max_rows', 1000000)


#from fancyimpute import KNN 

#读入数据
train = pd.read_csv('./remastered.csv')
test = pd.read_csv('./test_prepro.csv')

#log_y
train_target = train['SalePrice']

#去废列
train = train.drop(['index','Id'],axis = 1)
test = test.drop(['index','Id'],axis = 1)

# 特征工程
def transform(X):
    X.drop('SaleCondition',axis=1,inplace=True)
    X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
    X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
    X['Yrsold_minus_Remod'] = X['YrSold']-X['YearRemodAdd']
    #X[X['Yrsold_minus_Remod'] < 0] = 0.9
    X.loc[X['Yrsold_minus_Remod']<0,'Yrsold_minus_Remod']=0.9
    X['Yrsold_minus_garage'] = X['YrSold']-X['GarageYrBlt']
    X.loc[X['Yrsold_minus_garage']<0,'Yrsold_minus_garage']=0.9
    X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
    X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]

    
    X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]

    X["-_Functional_TotalHouse"] = X["Functional"] * X["TotalHouse"]
    X["-_Functional_OverallQual"] = X["Functional"] + X["OverallQual"]
    X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
    X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]



    X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
    X["Rooms"] = X["FullBath"]+X["TotRmsAbvGrd"]
    X["PorchArea"] = X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]
    X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]


    return X


fullfill_train = pd.read_csv('./fullfill_train.csv')
fullfill_test = pd.read_csv('./fullfill_test.csv')


fullfill_train['SalePrice'] = train_target

fullfill_train_X = fullfill_train.drop(['SalePrice'],axis = 1)



train_without4_without_price = train.drop(['LotFrontage','MasVnrArea','MasVnrType','Electrical','SalePrice'],axis=1)
test_without4 = test.drop(['LotFrontage','MasVnrArea','MasVnrType','Electrical'],axis=1)


train_with4 = pd.concat([train_without4_without_price,fullfill_train[['LotFrontage','MasVnrArea','MasVnrType','Electrical']]],axis = 1)
test_with4 = pd.concat([test_without4,fullfill_test[['LotFrontage','MasVnrArea','MasVnrType','Electrical']]],axis = 1)

#feature engeering
train_with4 = transform(train_with4)

test_with4 = transform(test_with4)


train_with4.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold','MoSold'],axis = 1,inplace=True)


test_with4.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold','MoSold'],axis = 1,inplace=True)


#after FE, do skewness and one-hot for train
X_numeric=train_with4.select_dtypes(exclude=["object"])
skewness = X_numeric.apply(lambda x: skew(x))
skewness_features = skewness[abs(skewness) >= 0.75].index
train_with4[skewness_features] = np.log1p(train_with4[skewness_features])
train_with4 = pd.get_dummies(train_with4)

#after FE, do skewness and one-hot for test
X_numeric=test_with4.select_dtypes(exclude=["object"])
test_with4[skewness_features] = np.log1p(test_with4[skewness_features])
test_with4 = pd.get_dummies(test_with4)

#consistancy
full3 = pd.concat([train_with4,test_with4])
full3 = full3.fillna(0)
train_with4 = full3.iloc[:1460,:]
test_with4 = full3.iloc[1460:,:]


#PCA-scale
scaler = RobustScaler()
X_scaled_train = scaler.fit(train_with4).transform(train_with4)

X_scaled_test = scaler.fit(train_with4).transform(test_with4)


#log-price
log_train_target = np.log(train_target)

#pca 
pca = PCA(n_components=79)
train_after_pca_87 = pca.fit_transform(X_scaled_train)
test_after_pca_87 = pca.transform(X_scaled_test)


def choose_data_for_train(origin_data,data,log_y):
    '''
    description for input:
    {
    origin_data: 最原始数据；
    data： 用于模型训练的数据
    }
    description for output:
    {
    data for_training the model
    }
    '''
    origin_data = pd.DataFrame(origin_data)
    data = pd.DataFrame(data)
    #Find SaleCondition=='Normal' observations to fit the hyperparameters
    normal_index = origin_data[origin_data['SaleCondition']=='Normal'].index
    data_to_choose_hyperparas_x = data.ix[normal_index]
    data_to_choose_hyperparas_y = log_y.ix[normal_index]
    return data_to_choose_hyperparas_x,data_to_choose_hyperparas_y

data_to_choose_hyperparas_x,data_to_choose_hyperparas_y = choose_data_for_train(train,train_after_pca_87,log_train_target)
  
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        #print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
'''
#svr
params = {
        'gamma' : [x/10000 for x in np.arange(1,20,1)],
        #'C' : [np.arange(10,16)],
        'epsilon' : [x/1000 for x in np.arange(1,10,1)]
        }   
'''
'''
#lasso
params = {
        'alpha':[x/10000 for x in np.arange(1,20,1)],
        'max_iter':[x*10000 for x in np.arange(1,5,1)]
        }  
'''
'''
#en
params = {
        'alpha':[x/1000 for x in np.arange(1,20,1)],
        'l1_ratio':[x/100 for x in np.arange(1,20,1)],
        'max_iter':[x*10000 for x in np.arange(1,5,1)]
        } 
'''
'''
#kr
params = {
        'alpha':[x/10 for x in np.arange(1,20,1)],
        'degree':[x for x in np.arange(1,5,1)],
        'coef0':[x/10 for x in np.arange(1,10,1)]
        }
'''
'''
#rf
params = {
        'n_estimators':range(10,91,10)
        }
#grid(SVR(kernel='rbf',C=13)).grid_get(data_to_choose_hyperparas_x,data_to_choose_hyperparas_y,params)
#grid(Lasso()).grid_get(data_to_choose_hyperparas_x,data_to_choose_hyperparas_y,params)
#grid(KernelRidge(kernel='polynomial')).grid_get(data_to_choose_hyperparas_x,data_to_choose_hyperparas_y,params)
grid(RandomForestRegressor()).grid_get(data_to_choose_hyperparas_x,data_to_choose_hyperparas_y,params)
'''

#svr = SVR(kernel='rbf',C=13, epsilon=0.009 , gamma=0.0011)
#lasso = Lasso(alpha=0.0001,max_iter=10000)
#kr = ElasticNet(alpha=0.005,l1_ratio=0.01,max_iter=10000)
#kr = KernelRidge(kernel='polynomial',alpha=0.4,coef0=0.9,degree=2)
rf = RandomForestRegressor(n_estimators=70)
model = rf
model.fit(data_to_choose_hyperparas_x,data_to_choose_hyperparas_y)
#model1 = svr
#model2 = kr
#model1.fit(data_to_choose_hyperparas_x,data_to_choose_hyperparas_y)
#model2.fit(data_to_choose_hyperparas_x,data_to_choose_hyperparas_y)

def alpha_calculation(price_origin_train, X_post_PCA, imputed_data, listype, model):
    '''
    input{
    price_origin_train:remastered.csv;
    X_post_PCA: data after PCA;
    imputed_data:data containing the column SalePrice;
    listype:the value of SaleCondition;
    model: trained model
    }
    '''
    #assign values to X,y:
    #data_X = imputed_data.drop('SalePrice',axis = 1)
    data_y = imputed_data['SalePrice']
    log_y = np.log(data_y)
    
    #PCA transformation:
    #redo_data_X = scaler.fit(data_X).transform(data_X)
    #trans_redo_data_X = pca.fit_transform(redo_data_X)
    
    #find index 
    index = list(price_origin_train[price_origin_train['SaleCondition'] == listype].index)
    #print(index)
    
    #find the relevant values corresponding to the index
    trans_redo_data_X_dataframe = pd.DataFrame(X_post_PCA).ix[index]
    
    #print(remastered_train,train_X_after_PCA, train_data)
    
    #prediction
    pred = model.predict(trans_redo_data_X_dataframe.values)
    
    #find the alpha corresponding to the abnormal salescondition
    attribute_alpha = log_y.ix[index].values.T/pred
    
    alpha = attribute_alpha.mean()
    origin_mse = mean_squared_error(log_y.ix[index].values.T,pred.reshape(-1,1))
    modified_mse = mean_squared_error(log_y.ix[index].values.T,alpha*pred.reshape(-1,1))
    ratio = (origin_mse-modified_mse)/origin_mse
    print('*'*50)
    print(listype)
    print('origin_mse','||||modified_mse||||','ratio of improvement')
    print(origin_mse,modified_mse,ratio)
    #return alpha,origin_mse,modified_mse,ratio,model
    return alpha

sale_condition = ['Abnorml','AdjLand','Alloca','Family','Partial']
alpha_dic = {
        'Normal':1,
        'Abnorml':0,
        'AdjLand':0,
        'Alloca':0,
        'Family':0,
        'Partial':0
        }

for each in sale_condition:
    alpha_dic[each] = alpha_calculation(train,train_after_pca_87,fullfill_train,each,model)
'''
alpha_dic2 = {
        'Normal':1,
        'Abnorml':0,
        'AdjLand':0,
        'Alloca':0,
        'Family':0,
        'Partial':0
        }
for each in sale_condition:
    alpha_dic2[each] = alpha_calculation(train,train_after_pca_87,fullfill_train,each,model2)
'''
#fa_alpha,fa_origin_mse,fa_modified_mse,fa_ratio,model = alpha_calculation(train,train_after_pca_87, fullfill_train,'Family',xgboost)
def predict_model(input_data,origin_data,model,alpha_dic):
    '''
    input{
    input_data: data to be predicted;
    origin_data: data containing saleCondition
    }
    '''
    pret_without_alpha = model.predict(input_data)
    pred_with_alpha = []
    for i in range(0,len(test['SaleCondition'])):
        pred_with_alpha.append(pret_without_alpha[i]*alpha_dic[test['SaleCondition'][i]])
    #return np.array(pred_with_alpha)
    return pret_without_alpha

res = predict_model(test_after_pca_87,train,model,alpha_dic)
#res1 = predict_model(test_after_pca_87,train,model1,alpha_dic)
#res2 = predict_model(test_after_pca_87,train,model2,alpha_dic2)
#res = (res1+res2)/2.0
#res = predict_model(test_after_pca_87,train,model)
predition = pd.DataFrame(data = np.exp(res),columns = ['SalePrice'],index = range(1461,2920))
predition.index.name = 'Id'
predition.to_csv('./predition.csv',index = True)


