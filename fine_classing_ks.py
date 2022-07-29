#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:44:45 2022

@author: parkkwonsoo
@title: CPS variables fine classing
 
"""
import pandas as pd
import numpy as np
import time


def ext_indx(train, test, target, column_name, min_bound, max_bound):
#==============================================================================
# ext_indx : extract indexes(IV, PSI etc..) of column
#   1. train        : input train dataset (made by "class_grp" function)
#   2. test         : input test dataset (made by "class_grp" function)
#   3. target       : y column name (text)
#   4. column_name  : variable from which to extract indexes (text)
#   5. min_bound    : class boundary of minimum (made by "class_grp" function)
#   6. max_bound    : class boundary of maximum (made by "class_grp" function)
#==============================================================================
    # initialize
    CLASS1 = []
    G1 = []
    PG1 = []
    B1 = []
    PB1 = []
    I1 = []
    PI1 = []
    T = []
    PT = []
    GRATE1 = []
    BRATE1 = []
    TODDS1 = []
    IODDS1 = []
    GBIDX1 = []
    CPGB_diff = []
    AUROC = []
    IV = []
    CPG = 0
    CPB = 0
    PSI = 0
    KS1 = ['']*len(min_bound)
    GINI1 = ['']*len(min_bound)
    IV1 = ['']*len(min_bound)
    PSI1 = ['']*len(min_bound)
    
    # target condition with train set
    good_cond = (train[target]==0)
    bad_cond = (train[target]==1)
    und_cond = (train[target]==2)
    
    GT = train[good_cond].count()[target]
    BT = train[bad_cond].count()[target]
    IT = train[und_cond].count()[target]
    TT = train.count()[target]
    test_TT = test.count()[target]
    
    for i in range(0, len(min_bound)):
        mini, maxi = min_bound[i], max_bound[i]
        bet_cond = (train[column_name].between(mini, maxi))
        Gv, Bv, Iv, Tv = train[bet_cond & good_cond].count()[0], train[bet_cond & bad_cond].count()[0], train[bet_cond & und_cond].count()[0], train[bet_cond].count()[0]
        PGv, PBv, PIv, PTv = Gv/GT, Bv/BT, Iv/IT, Tv/TT
        test_PTv = test[(test[column_name].between(mini, maxi))].count()[0] / test_TT
        CLASS1.append(i+1)
        G1.append(Gv)
        PG1.append(PGv)
        B1.append(Bv)
        PB1.append(PBv)
        I1.append(Iv)
        PI1.append(PIv)
        T.append(Tv)
        PT.append(PTv)
        GRATE1.append(Gv/Tv)
        BRATE1.append(Bv/Tv)
        if Bv == 0:
            TODDS1.append(0)
        else:
            TODDS1.append(Gv/Bv)
        if PGv >= PBv:
            if PBv == 0:
                IODDS1.append(0)
                GBIDX1.append('100G')
            else:
                IODDS1.append(PGv/PBv)
                GBIDX1.append(str(round(PGv/PBv*100))+'G')
        else:
            if PGv == 0:
                IODDS1.append(0)
                GBIDX1.append('100B')
            else:
                IODDS1.append(2 - PBv/PGv)
                GBIDX1.append(str(round(PBv/PGv*100))+'B')
        CPG += PGv
        CPB += PBv
        CPGB_diff.append(abs(CPG-CPB))
        AUROC.append(0.5*PGv*PBv+(1-CPG)*PBv)
        if (PGv == 0)|(PBv == 0):
            IV.append(0)
        else:
            IV.append((PGv-PBv)*np.log(PGv/PBv))
        np.seterr(divide='ignore')
        PSI += (test_PTv-PTv)*np.log(test_PTv/PTv)
        np.seterr(divide='warn')
    
    KS1[0] = max(CPGB_diff)
    GINI1[0] = 2*sum(AUROC)-1
    IV1[0] = sum(IV)
    PSI1[0] = PSI
    res_df = pd.DataFrame({'VAR':column_name, 'CLASS': CLASS1,'MIN':min_bound, 'MAX':max_bound, 'G1':G1, 'PG1':PG1, 'B1':B1, 'PB1':PB1, 'I1':I1, 'PI1':PI1, 'T':T, 'PT':PT,
                           'GRATE1':GRATE1, 'BRATE1':BRATE1, 'TODDS1':TODDS1, 'IODDS1':IODDS1, 'GBIDX1':GBIDX1, 'KS1':KS1, 'GINI1':GINI1, 'IV1':IV1, 'PSI1':PSI1})
    return res_df



def class_grp(inp_df, target, column_name, val_flag, grp=20):
#==============================================================================
# class_grp : column devided by specific group
#   1. inp_df       : input dataset
#   2. target       : y column name (text)
#   3. column_name  : variable from which to extract indexes (text)
#   4. val_flag     : train/test split flag (0:train, 1:test1, 2:test2 ...)
#   5. grp          : number of class (default=20, each class have 5%)
#==============================================================================
    data = inp_df[[target,column_name, val_flag]]
    
    # Special Values
    SV_list = [9999999.9, 8888888.8, 999999999, 888888888]
    
    # extract global min, max
    min_val = [data[column_name].min()]
    if data[column_name].isin(SV_list).sum() > 0:
        max_val = [data.loc[~(data[column_name].isin(SV_list))][column_name].max()]
    else:
        max_val = [data[column_name].max()]
    
    # train, test split
    train = data[data[val_flag]==0]
    # multiple testset
    test_no = len(data[val_flag].unique())
    if test_no > 2:
        for i in range(0, test_no):
            globals()['test%s'%(i+1)] = data[data[val_flag]==(i+1)]
    else:
        test1 = data[data[val_flag]==1]
        test2 = data[data[val_flag]==2]
        
    # classing
    dist = 1/grp
    quant = np.arange(0, 1, dist).tolist()
    pctl = train.quantile(q=quant, axis=0, numeric_only=True)
    max_val2 = pctl[column_name].drop_duplicates().tolist()
    max_val2 = [i for i in max_val2 if i not in SV_list]
    if max_val2[-1] == max_val[0]:
        max_val2 = max_val2[:-1]
    else:
        pass
    SV_class = []
    if data[column_name].isin(SV_list).sum() > 0:
        for i in range(0, len(SV_list)):
            if data[column_name].isin([SV_list[i]]).sum() > 0:
                SV_class.append(SV_list[i])
            else:
                pass
        max_val2 = max_val2[1:]
    else:
        pass

    min_val2 = [i+0.1 for i in max_val2]
    
    min_bound = SV_class + min_val + min_val2
    max_bound = SV_class + max_val2 + max_val
    
    res = ext_indx(train, test1, target, column_name, min_bound, max_bound)
    

    if ((res["MIN"][0]<=0) & (res["MAX"][0]<=0) & (res["PT"][0]<0.05)):
        new_min_bound = min_val + min_bound[2:]
        new_max_bound = max_bound[1:]
         
        res = ext_indx(train, test1, target, column_name, new_min_bound, new_max_bound)
    else:
        pass

    
    return res
    

def fine_classing(inp_df, target, column_list, val_flag, grp=20):
#==============================================================================
# fine_classing : fine classing and extract indexes of column list
#   1. inp_df       : input dataset
#   2. target       : y column name (text)
#   3. column_list  : variables from which to extract indexes (list)
#   4. val_flag     : train/test split flag (0:train, 1:test1, 2:test2 ...)
#   5. grp          : number of class (default=20, each class have 5%)
#==============================================================================
    start_time = time.time()
    data = inp_df[column_list + [target, val_flag]]
    
    res_concat = pd.DataFrame([])
    list_total = len(column_list)
    for i in column_list:
        res = class_grp(data, target, i, val_flag, grp=20)
        res_concat = pd.concat([res_concat, res])
        print(" {vname} ({now} / {total}) classing complete... ".format(vname=i, now=column_list.index(i) + 1, total=list_total))
        
    runtime = round(time.time() - start_time, 3)
    print("===== Running Time : {rt} seconds ======".format(rt=runtime))
    return res_concat
    