import pandas as pd
import glob
from combat.pycombat import pycombat
import pandas as pd
import matplotlib.pyplot as plt
# combat simdilik calismiyor
def three_class_prediction(
                           No_fea_sel=False,
                           normalization_bool=True
                           
                           ):

    import pandas as pd
    import glob
    data_list = None
    survival_file = None
        # Brats data
    data_list = sorted(glob.glob('E:/scripts_kerim/radiomics/ml_class/rad_fet_/batch_results/manual/t2_64/batch/combined_results.csv'))
    
    # Storm data
    # data_list2 = sorted(glob.glob('/home/kerimduman/Downloads/radiomics/ml_class/rad_fet_/batch_results/manual/t2s_64/batch/combined_results.csv'))
    
    # Storm data resampled
    data_list2 = sorted(glob.glob('E:/scripts_kerim/radiomics/ml_class/rad_fet_/batch_results/manual/t2s_64/batch/combined_results.csv'))    
    
    
        # nnunet
    # data_list2=sorted(glob.glob('/home/kerimduman/Downloads/radiomics/ml_class/rad_fet_/batch_results/auto_nnunet/t2s_auto_64/batch/combined_results.csv'))
        # wu rfs+
    # data_list2=sorted(glob.glob('/home/kerimduman/Downloads/radiomics/ml_class/rad_fet_/batch_results/auto_wurfs_bin/t2s_auto_64/batch/combined_results.csv'))
        #(general purpose)  wu rfs+ edited p_Val 
    # data_list2=sorted(glob.glob('/home/kerimduman/Downloads/radiomics/spaarc/batch_results/t2s_auto_64/batch/combined_results.csv'))

        # 1 mod 

    # data_list2=sorted(glob.glob('/home/kerimduman/Downloads/radiomics/ml_class/rad_fet_/batch_results/auto_1mod/1mod_t1ce/t2s_auto_64/batch/combined_results.csv'))    
    
    #    # rfs
    # data_list2=sorted(glob.glob('/home/kerimduman/Downloads/radiomics/spaarc/batch_results/t2s_auto_64_u-rfs/batch/combined_results.csv'))    
    
       # mclass
    # data_list2=sorted(glob.glob('/home/kerimduman/Downloads/radiomics/spaarc/batch_results/t2s_auto_64_4mod_binary/batch/combined_results.csv'))    
    
    # burdenko resample
    data_list3=sorted(glob.glob(r'E:/scripts_kerim/radiomics/spaarc/feature_extraction/batch_results/t2_no_int_z_norm/t2b_64/batch/combined_results.csv'))    


    # 1brats 2020
    survival_file = 'E:/scripts_kerim/radiomics/ml_class/survival_info.csv'
    # storm
    survival_file2 = 'E:/scripts_kerim/radiomics/ml_class/os_surv_storm.csv'
    # burdenko
    survival_file3 = 'E:/kerim_backup/burdenko/output_burdenko/burdenko_clinical_edited.csv'

    data_tc = pd.read_csv(data_list[0])
    data_tc2 = pd.read_csv(data_list2[0])
    data_tc3 = pd.read_csv(data_list3[0])

    # Load survival info
    data_survival = pd.read_csv(survival_file)
    data_survival2 = pd.read_csv(survival_file2)
    data_survival3 = pd.read_csv(survival_file3)


    
    
    
    
    left=data_tc
    left.head()
    
    left2=data_tc2
    left2.head()
    
    left3=data_tc3
    left3.head()
    
    right=data_survival.copy()
    
    right.head()
    
    right.rename(columns = {'Brats20ID':'PatientID' }, inplace = True)
    
    right.head()
    
    # second part
    right2=data_survival2.copy()
    right2 = right2[['PatientID','Age', 'OS']]
    right2.head()
    
    right3=data_survival3.copy()
    right3 = right3[['patient_name', 'age', 'OS']]
    right3.head()
    
    right3.rename(columns = {'patient_name':'PatientID' }, inplace = True)
    right3.rename(columns = {'age':'Age' }, inplace = True)


    
    import numpy as np
    import os

    for root, dirs, files in os.walk('E:/scripts_kerim/monai/the last version - sripped and registered', True):
       break
    
    dirs.sort()
    
    import numpy as np
    import os
    
    for rootb, dirsb, filesb in os.walk('E:/scripts_kerim/monai/burdenko_ext_nn_rename_segmentation/nifti/', True):
       break
    
    dirsb.sort()


    # Define two NumPy arrays of strings
    array1 = np.array([left2['PatientID']])
    array2 = np.array([dirs])
    
    # Find the differences between the two arrays
    difference1 = np.setdiff1d(array2, array1)
    
    
    
    
    # merging data with survival info only 236
    result = pd.merge(left, right, on="PatientID")
    result.head()
    
    # did not use this used below
    # result2 = pd.merge(left2, right2, on="PatientID")
    # how=right means keep 
    result2 = pd.merge(left2, right2, on="PatientID",how="right")
    
    result2.head()
    # add patients without segmentations
    filtered_df = result2[result2['PatientID'].isin(dirs)]
    result2=filtered_df
    
    result3 = pd.merge(left3, right3, on="PatientID",how="right")

    result3.head()
    # add patients without segmentations
    filtered_df2 = result3[result3['PatientID'].isin(dirsb)]
    result3=filtered_df2
    
    
    
    # imputing missing values
    df=result2
    # df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)
    # stgl 68 similar to 67 and copied to 68
    # df.loc[54, df.columns[1:¡-2]] = df.loc[53, df.columns[1:-2]].tolist()
    
    result2=df
    
    # imputing missing values burdenko
    df2=result3
    # df.fillna(df.mean(), inplace=True)
    df2.fillna(0, inplace=True)
    # stgl 68 similar to 67 and copied to 68
    # df.loc[54, df.columns[1:¡-2]] = df.loc[53, df.columns[1:-2]].tolist()
    
    result3=df2
    
    
    
    
    

    #line 83 has no exact number patients still alive
    # alive live long 365+
    result=result.drop([83])
    

    
    
    import numpy as np
    
    from  sklearn.linear_model import Lasso
    
    from sklearn.preprocessing import StandardScaler
    
    from sklearn.pipeline import Pipeline
    
    from sklearn.model_selection import GridSearchCV, train_test_split
    
    from sklearn.datasets import load_diabetes
    
    X=result.iloc[:,1:-2]
    y=result.iloc[:,-2]
    y = y.astype({'Survival_days':'float16'})
    
    

    
    X2=result2.iloc[:,1:-1]
    y2=result2.iloc[:,-1]
    y2 = y2.astype({'OS':'float16'})
    
    X3=result3.iloc[:,1:-1]
    y3=result3.iloc[:,-1]
    y3 = y3.astype({'OS':'float16'})
    
    # import libraries
    import pandas as pd
    import matplotlib.pyplot as plt
    

    
    # # # 
    # bootstrap
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42 )
    
    
    coef, selected_features,selected_features_mask = feature_selection_with_lasso(X_train, X_test, y_train, y_test,X, y)
    
    
    # No_fea_sel=True
    if No_fea_sel==True:
        # no feature selection
        features=X.columns.tolist()
        selected_features=np.array(features)
    

    selected_result_all=result[selected_features]
    
    selected_result_train_X=X_train[selected_features]
    
    selected_result_train_xy=pd.concat([selected_result_train_X, y_train], axis=1)
    
    
    #corelation results
    corelation=selected_result_all.corr()
    
    print(corelation)
    
    corelation2=selected_result_train_X.corr()
    
    corelation3=selected_result_train_xy.corr()
    
        
    # test setine uygulanısı
    from sklearn.preprocessing import StandardScaler
    
    sc1 = StandardScaler()
    # x train e göre ölçeklendi x test bu ölçeğe uygulanacak
    x_olcekli = sc1.fit_transform(selected_result_train_X)
    
    
    selected_result_test_X=X_test[selected_features]
    
    x_olcekli_test = sc1.transform(selected_result_test_X)
    
        
    
    selected_result_storm=X2[selected_features]
    
    selected_result_burd=X3[selected_features]
    import numpy as np
    # same 1 mod gives error due to infinite values 
    selected_result_burd = selected_result_burd.replace([np.inf, -np.inf], 0)
    
    
    x_olcekli_burd = sc1.transform(selected_result_burd)
    
    
    import numpy as np
    # same 1 mod gives error due to infinite values 
    selected_result_storm = selected_result_storm.replace([np.inf, -np.inf], 0)

    
    x_olcekli_storm = sc1.transform(selected_result_storm)
    
    # in month
    y_train2=y_train/30
    
    y3_burd=y3/30
    
    # label less and equal to shor and more as long survival as 0 (short) and 1 (long)
    
    import statistics
    
    # long short threshold 11
    # label=[int(item) for item in (y_train2>11)]
    import numpy as np
    
    # Assuming y_train2 is your original labels
    threshold_0 = 15
    threshold_1 = 10
    
    # Create binary labels based on thresholds using list comprehension
    # label = np.where(y_train2 > threshold_0, 0, np.where(y_train2 >= threshold_1, 1, 2))
    
    # prediction exactos month
    label=y_train2
    
    
    
    y_test2=y_test/30
    

    
    # exact month
    label2=y_test2
        

    # exact month
    label3=y2
    
    label4=y3_burd
    
    normalizaton_bool=True
    a1,a2,a3,aaaaxxx,med,selected_features= RRS_eval_bootsrap(X_train, X_test, y_train, y_test, X,y, X2,X3, x_olcekli,x_olcekli_test, x_olcekli_storm,x_olcekli_burd, y_train2, y_test2, y2,y3_burd)
    
  

    
    return a1, a2, a3,med,selected_features
        



import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

def feature_selection_with_lasso(X_train, X_test, y_train, y_test, X, y):
        
         
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Lasso
    features = X.columns.tolist()
    from sklearn.model_selection import RepeatedKFold

 

    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)


    pipeline = Pipeline([('scaler', StandardScaler()),
                          ('model', Lasso())])
    
    
    # type 1 alpha range
    search = GridSearchCV(pipeline,
                          {'model__alpha': np.arange(0.1, 3, 0.1)},
                          cv=rkf,
                          scoring='explained_variance',
                          verbose=3)


    search.fit(X_train, y_train)

    best_params = search.best_params_

    coef = search.best_estimator_.named_steps['model'].coef_

    # Ascending ranks, select top 15
    rank_coef = np.sort(coef)[::-1]
    
  
    
    

    # Selected features
    selected_features = np.array(features)[coef != 0]
    # selected_features_mask=[selected_features==selected_features]
    
    from sklearn.datasets import load_digits
    from sklearn.feature_selection import SelectKBest,mutual_info_regression, chi2,f_regression
    selected_result_train_X=X_train[selected_features]
    selected_result_train_X.shape
    from sklearn.preprocessing import MinMaxScaler
    
     
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(selected_result_train_X)
    print(normalized_data)
    
    # selector = SelectKBest(f_regression, k=len(selected_features))
    selector = SelectKBest(f_regression, k=3)
    X_train_new = selector.fit_transform(normalized_data, y_train)
    
    # Get the Boolean mask for selected features
    selected_features_mask = selector.get_support()
    
    yedek=selected_features
    yedek=coef
    selected_features=selected_features[selected_features_mask]
    
    

    return  coef, selected_features,selected_features_mask


import numpy as np
# import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR



def RRS_eval_bootsrap(X_train, X_test, y_train, y_test,X,y, X2,X3, x_olcekli,x_olcekli_test, x_olcekli_storm,x_olcekli_burd, y_train2, y_test2, y2, y3_burd,
                      normalizaton_bool=True, No_fea_sel=False):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    
    
    coef, selected_features,selected_features_mask  = feature_selection_with_lasso(X_train, X_test, y_train, y_test, X, y)
    
    
    
    # No_fea_sel=True
    if No_fea_sel==True:
        # no feature selection
        features=X.columns.tolist()
        selected_features=np.array(features)
    
    
    
    selected_result_train_X=X_train[selected_features]
    
    selected_result_train_xy=pd.concat([selected_result_train_X, y_train], axis=1)
    
    

    
    corelation2=selected_result_train_X.corr()
    
    corelation3=selected_result_train_xy.corr()
    
        
    # test setine uygulanısı
    from sklearn.preprocessing import StandardScaler
    
    sc1 = StandardScaler()
    # x train e göre ölçeklendi x test bu ölçeğe uygulanacak
    x_olcekli = sc1.fit_transform(selected_result_train_X)
    
    import matplotlib
    # train rss calculation
    aaaaxxx=coef[coef != 0]
    aaaaxxx = aaaaxxx[selected_features_mask]
    RSS_train = aaaaxxx * x_olcekli
    
    import numpy as np
    RSS_one=np.sum(RSS_train,axis=1)
   
    # do I need to scale this value too?
    sc2 = StandardScaler()
    # x train e göre ölçeklendi x test bu ölçeğe uygulanacak
    RSS_one2=RSS_one.reshape(-1,1)
    RSS_olcekli = sc2.fit_transform(RSS_one2)
    
    import matplotlib.pyplot as plt
    plt.scatter(y_train2,RSS_olcekli)
    plt.xlabel("OS")
    plt.ylabel("RSS")
   
   
   
    import statistics
    med=statistics.median(RSS_olcekli)
    
    # # Bootstrap the test set
    # from lifelines.utils import concordance_index
    # n_bootstrap_samples = 1000
    # bootstrap_mse = []

    # for _ in range(n_bootstrap_samples):
    #     # Sample with replacement from X_test and y_test
    #     indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
    #     X_test_sample = X_test.astype(np.float64).iloc[indices]
    #     y_test_sample = y_test.values.astype(np.float64)[indices]
        
    #     selected_result_test_X=X_test_sample[selected_features]
        
    #     x_olcekli_test = sc1.transform(selected_result_test_X)
        
    #     # train rss calculation
    #     RSS_test = aaaaxxx * x_olcekli_test
        
    #     RSS_two=np.sum(RSS_test,axis=1)
        
    #     RSS_two2=RSS_two.reshape(-1,1)
    #     RSS_olcekli2 = sc2.transform(RSS_two2)
        
    #     label2=[int(item) for item in (RSS_olcekli2<med)]
    #     # Predict and compute MSE
    #     # y_pred = model.predict(x_olcekli_test)
    #     y_pred = label2
        
    #     # Compute the C-index
    #     c_index_boot = concordance_index(y_test2, RSS_olcekli2)
    #     print("C-index bootstrap on test data:", c_index_boot)
        
    #     # mse = mean_squared_error(y_test_sample, c_index_boot)
    #     # bootstrap_mse.append(mse)

    #     bootstrap_mse.append(c_index_boot)

    # # Analyze bootstrap results
    # bootstrap_mse = np.array(bootstrap_mse)
    # mean_mse = bootstrap_mse.mean()
    # lower_bound = np.percentile(bootstrap_mse, 2.5)
    # upper_bound = np.percentile(bootstrap_mse, 97.5)

    

    
    
    selected_result_test_X=X_test[selected_features]

    x_olcekli_test = sc1.transform(selected_result_test_X)


    selected_result_storm=X2[selected_features]
    
    import numpy as np
    # same 1 mod gives error due to infinite values 
    selected_result_storm = selected_result_storm.replace([np.inf, -np.inf], 0)


    x_olcekli_storm = sc1.transform(selected_result_storm)
    
    selected_result_burd=X3[selected_features]

    import numpy as np
    # same 1 mod gives error due to infinite values 
    selected_result_burd = selected_result_burd.replace([np.inf, -np.inf], 0)
    
    
    x_olcekli_burd = sc1.transform(selected_result_burd)
        
    
    # train rss calculation
    RSS_train = aaaaxxx * x_olcekli

    RSS_one=np.sum(RSS_train,axis=1)

    # do I need to scale this value too?
    sc2 = StandardScaler()
    # x train e göre ölçeklendi x test bu ölçeğe uygulanacak
    RSS_one2=RSS_one.reshape(-1,1)
    RSS_olcekli = sc2.fit_transform(RSS_one2)

    y_train2=y_train/30

    import matplotlib.pyplot as plt
    plt.scatter(y_train2,RSS_olcekli)
    plt.xlabel("OS")
    plt.ylabel("RSS")



    import statistics
    med=statistics.median(RSS_olcekli)

    # train data for median RSS low- high stratification

    label=[int(item) for item in (RSS_olcekli<med)]

    colors = ['red','blue']


    plt.scatter(y_train2, RSS_olcekli, c=label, cmap=matplotlib.colors.ListedColormap(colors))


    # # #  validate with test dataset
    # train rss calculation
    RSS_test = aaaaxxx * x_olcekli_test

    RSS_two=np.sum(RSS_test,axis=1)



    # storm rss
    RSS_storm = aaaaxxx * x_olcekli_storm

    RSS_three=np.sum(RSS_storm,axis=1)
    
    # burdenko rss
    RSS_burd = aaaaxxx * x_olcekli_burd
    
    RSS_four=np.sum(RSS_burd,axis=1)



    if normalizaton_bool==True:
        # # x train e göre ölçeklendi x test bu ölçeğe uygulanacak
        RSS_two2=RSS_two.reshape(-1,1)
        RSS_olcekli2 = sc2.transform(RSS_two2)
        # x traine gore storm olceklendi
        RSS_three2=RSS_three.reshape(-1,1)
        RSS_olcekli3 = sc2.transform(RSS_three2)
        
        # x traine gore burdenko olceklendi
        RSS_four2=RSS_four.reshape(-1,1)
        RSS_olcekli4 = sc2.transform(RSS_four2)

    
    # normalizasyonsuz
    if normalizaton_bool==False:
        # part2
        RSS_two2=RSS_two.reshape(-1,1)
        RSS_olcekli2 = RSS_two
        
        RSS_three2=RSS_three.reshape(-1,1)
        RSS_olcekli3 = RSS_three
        
        RSS_four2=RSS_four.reshape(-1,1)
        RSS_olcekli4 = RSS_four
        
        
    label2=[int(item) for item in (RSS_olcekli2<med)]
    colors = ['red','blue']
   
    plt.show()
    plt.scatter(y_test2, RSS_olcekli2, c=label2, cmap=matplotlib.colors.ListedColormap(colors))
   
   
    # storm y2 no need to convert to month
   
    label3=[int(item) for item in (RSS_olcekli3<med)]
    colors = ['red','blue']
   
    plt.show()
    plt.scatter(y2, RSS_olcekli3, c=label3, cmap=matplotlib.colors.ListedColormap(colors))
    
    
    plt.show()
    # plt.show()
    
   

    ##train results
    os1=y_train2[np.array(label)==0]
    os2=y_train2[np.array(label)==1]
    
    #km visaulisation
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    
    ax = plt.subplot(111)
    ax = kmf.fit(os1,  label="Low Risk").plot(ax=ax,ci_show=False)
    ax = kmf.fit(os2,  label="High Risk").plot(ax=ax,ci_show=False)
    # Label the axes
    ax.set_xlabel("OS")
    ax.set_ylabel("The probability of survival")
    plt.show()
    
    #logrank_test / chi square
    from lifelines.statistics import logrank_test
    results=logrank_test(os1,os2)
    results.print_summary()
    results.p_value
    train_p=results.p_value
    
    ##test results
    os1=y_test2[np.array(label2)==0]
    os2=y_test2[np.array(label2)==1]
    
    #km visaulisation
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    
    ax = plt.subplot(111)
    ax = kmf.fit(os1,  label="Low Risk").plot(ax=ax,ci_show=False)
    ax = kmf.fit(os2,  label="High Risk").plot(ax=ax,ci_show=False)
    # Label the axes
    ax.set_xlabel("OS")
    ax.set_ylabel("The probability of survival")
    
    #logrank_test / chi square
    from lifelines.statistics import logrank_test
    results=logrank_test(os1,os2)
    results.print_summary()
    results.p_value
    test_p=results.p_value
    
    
    plt.show()
    ##storm results
    os1=y2[np.array(label3)==0]
    os2=y2[np.array(label3)==1]
    
    #km visaulisation
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    
    # ax = plt.subplot(111)
    # ax = kmf.fit(os1,  label="Group 1-Treatment").plot(ax=ax,ci_show=False)
    # ax = kmf.fit(os2,  label="Group 2-Treatment").plot(ax=ax,ci_show=False)
    
    ax = plt.subplot(111)
    ax = kmf.fit(os1,  label="Low Risk").plot(ax=ax,ci_show=False)
    ax = kmf.fit(os2,  label="High Risk").plot(ax=ax,ci_show=False)
    # Label the axes
    ax.set_xlabel("OS")
    ax.set_ylabel("The probability of survival")
    #logrank_test / chi square
    from lifelines.statistics import logrank_test
    results=logrank_test(os1,os2)
    results.print_summary()
    results.p_value
    storm_p=results.p_value
    
    plt.show()

    
    
    print('train : {} and test: {} and storm: {} '.format(train_p,test_p,storm_p,))
    print('train : {} and test: {} and storm: {} '.format(train_p<0.05,test_p<0.05,storm_p<0.05))
    
    import pandas as pd
    from lifelines.datasets import load_rossi
    from lifelines.utils import concordance_index
    from lifelines import CoxPHFitter
    from sklearn.model_selection import train_test_split
    

    
   
    return train_p,test_p,storm_p,aaaaxxx,med,selected_features
   
# Example usage:

P_val_train,P_val_test,P_val_Storm,RRS_threshold,selected_features=three_class_prediction()
