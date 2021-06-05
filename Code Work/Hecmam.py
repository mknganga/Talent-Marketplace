# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:20:19 2020

@author: Martin Nganga
"""

import pandas as pd
import numpy as np
import string
import re
from itertools import chain
import sys
import pyomo.environ as pyo
import math


data = pd.read_csv('data.csv',index_col=0) # this is the new data received from Maj Doherty 
data.fillna(0,inplace=True)

## Creating a Preference Data Frame and saving it 

pref_df = data.iloc[:,8:]


# MCC Dataframe 
mcc_df = data.drop(['AMOS1','AMOS2','AMOS3','AMOS4','AMOS5','PMOS'], axis=1)


########################################## Processing for the MOS-Qualification Dataframe ###########################################################################################

M_PMOS = data.drop(['MCC','MCC2'], axis = 1)

job_list = list(M_PMOS.iloc[:,6:].columns) # holding the list of columns 


## Grabbing the BMOS on each column 
col_list = []
for i in job_list:
    colnum = re.findall('\d{4}', i )
    col_list.append(colnum)

col_list = list(chain.from_iterable(col_list))

M_PMOS.columns = M_PMOS.columns[:6].tolist() + col_list



map_df = pd.read_csv('Map_MOS.csv',index_col=0) 

d = {} # create a dictionary for mapping MOS to BMOS
for line in open("Map_MOS.csv").readlines():
    line = line.strip()
    line = line.split(",")
    key = line[0]
    for value in line[1:]:
        d.setdefault(key, []).append(value)
#print(d)


a = []
for i in M_PMOS.index:
    row = []
    for j in M_PMOS.iloc[:,6:].columns:
        values = d[j]
        values = list(map(int, values))
        if M_PMOS.loc[[i],['PMOS']].values[0] in values:
            row.append(1)
        elif M_PMOS.loc[[i],['AMOS1']].values[0] in values:
            row.append(1)
        elif M_PMOS.loc[[i],['AMOS2']].values[0] in values:
            row.append(1)
        elif M_PMOS.loc[[i],['AMOS3']].values[0] in values:
            row.append(1)
        elif M_PMOS.loc[[i],['AMOS4']].values[0] in values:
            row.append(1)    
        elif M_PMOS.loc[[i],['AMOS5']].values[0] in values:
            row.append(1)
        else:
            row.append(0)
    a.append(row)

mos_df = pd.DataFrame.from_records(a) 
mos_df.columns = job_list
mos_df.index = M_PMOS.index

## Create dictionaries for key is Marine and value is a list of jobs qualified

j_dic = {}
for i in mos_df.index:
    temp = []
    for j in mos_df.columns:
        if mos_df.at[i, j] == 1:
            temp.append(j)
    j_dic[i] = temp


## Create dictionaries for key is the job  and value is a list of Marines that qualify for that job

i_dic = {}
for j in mos_df.columns:
    temp = []
    for i in mos_df.index:
        if mos_df.at[i, j] == 1:
            temp.append(i)
        else:
            next
    i_dic[j] = temp
######################################## Processing for PCS Cost #########################################################################

# read data with MCC for Marines and Jobs 
# The JM_mcc CSV file was modified in excel for the columns to contain Jobs' MCC code only 
    
jb = pd.read_csv('JM_mcc.csv',index_col=0, header=0)
jb.columns = jb.columns.str.split('.').str[0]
jb.fillna(0,inplace=True)


mcc_zip = pd.read_csv('mcc_zip.csv')
mcc_zip.fillna(0,inplace=True)

# Merging Dataframes to capture each Marine's Zip code 

jb['UUID'] = jb.index
jb_new = jb.merge(mcc_zip, left_on='MCC',right_on= 'MCC', how='left')

jb_new = jb_new.drop_duplicates(subset='UUID', keep="first")

jb_new = jb_new.set_index("UUID")



mcc_zip1 = mcc_zip.drop_duplicates(subset='MCC', keep="first")

## Code to assign zip code to the jobs 

cols = jb_new.iloc[:,1:].columns.tolist()
m = []
no = []
for i in cols:
    i = str(i)
    for j in mcc_zip1['MCC'].values:
        j= str(j)
        if i == j:
            m.append(int(mcc_zip1.loc[mcc_zip1['MCC']==j, 'ZIPCODE'].values[0]))
            no.append(i)
            #print (i,j)
        else:
            next

zi = jb_new
zi = zi.drop('MCC', axis=1)
zi = zi.drop('ZIPCODE', axis=1)
zi.columns = m 
zi['ZIPCODE'] =jb_new['ZIPCODE']


############### Calculate Distance Using Great Circle #####################################################################################################33

geo = pd.read_csv('geo_data.csv')

# Merge dataframe to capture each Marine's geographical position 

zi['UUID'] = zi.index
calc = zi.merge(geo, left_on='ZIPCODE',right_on= 'zipcode', how='left')
calc = calc.set_index("UUID")
calc = calc.drop('ZIPCODE', axis=1)

geo_index = geo.set_index("zipcode")


# Function to calculate Distance 

def deg2rad(x):
    '''This function take three input integer and convert it into an output radian in float '''
    val = (x)*(math.pi/180)
    return val

def rad2nm(r):
    '''This function takes three input integer radian  and converts it to an output float nautical miles '''
    Value = r*((180*60)/math.pi)
    return Value

def calc_distance(a,b,c,d):
    '''This function takes four integers inputs and calculates the distance of two point in a great circle and give a distance output float'''
    a= deg2rad(a)
    b= deg2rad(b)
    c= deg2rad(c)
    d= deg2rad(d)
    cos1 = math.sin(a)*math.sin(c)+math.cos(a)*math.cos(c)*math.cos(b-d)
    cos1 = min(cos1, 1.0)
    value = rad2nm(math.acos(cos1))
    return value


# Actual Calculation 
    
'''
a = []
for i in calc.index:
    row = []
    lat1 = calc.loc[[i], ['lat']].values[0][0]
    lon1 = calc.loc[[i],['long']].values[0][0]
    for j in calc.iloc[:,:196].columns:
        zip_code = j
        for k in geo_index.index:
            if j == k:
                lat2 = geo_index.loc[[k],['lat']].values[0][0]
                lon2 = geo_index.loc[[k],['long']].values[0][0]
                r = calc_distance(lat1,lon1,lat2,lon2)
                if r >= 100:
                    cost = 2000 + (r*0.17)
                    row.append(cost)
                else:
                    cost = 0
                    row.append(cost)
            else:
                pass
    a.append(row)
    
cost_df = pd.DataFrame.from_records(a)

cost_df.index = mos_df.index
cost_df.columns = mos_df.columns'''   
    
# cost_df.to_csv("cost.csv") # You can save the cost file since it will not change once created 

#  We have already generated and created the cost data and stored it as cost.csv  (improve the speed of running the full code)
# Read in the cost data into a dataframe

cost_df = pd.read_csv('cost.csv',index_col=0)

############################################################################### PYOMO IMPlEMENTATION##############################################################################################

## HECMAM Model 

model = pyo.ConcreteModel()

I = list(pref_df.index)


J = []
for key in i_dic:
    if len (i_dic[key]) == 0:
        print("Warning: Job ",key," does not have any qualifying marine. It'll be ignored")
        next
    else:
       J.append(key) 
    
c = {(i, j):cost_df.loc[i,j] for i in I for j in j_dic[i]}
p = {(i, j):pref_df.loc[i,j] for i in I for j in j_dic[i]}
# w = 200


model.x = pyo.Var(I,J, within= pyo.Binary)
model.z_1 = pyo.Var(domain=pyo.Reals)
model.z_2 = pyo.Var(domain=pyo.Reals)


def one_job_per_marine(model,i):
    return sum(model.x[i,j] for j in j_dic[i]) == 1
model.one_job_per_marine = pyo.Constraint(I, rule=one_job_per_marine)


def marine_job(model,j):
    return sum(model.x[i,j] for i in i_dic[j]) <= 1
model.marine_job = pyo.Constraint(J, rule=marine_job)


def preference(model):
    return model.z_1 == sum(p[i,j]*model.x[i,j] for i in I for j in j_dic[i]) 
model.preference = pyo.Constraint(rule=preference)


def cost(model):
    return model.z_2 == sum(c[i,j]*model.x[i,j] for i in I for j in j_dic[i])  
model.cost = pyo.Constraint(rule=cost)


def obj_rule_z1(model):
    return model.z_1
model.obj = pyo.Objective(rule = obj_rule_z1,sense=pyo.maximize)

# Solve the model and report the results 
opt = pyo.SolverFactory("cbc")

results = opt.solve(model, tee=False)

## Display Results 

res_dic = {}
for i in I:
    res_dic[i]= {}
    for j in j_dic[i]:
        #print(model.x[i,j], pyo.value(model.x[i,j]))
        
        #res_dic[i][j] = pyo.value(model.x[i,j])
        if pyo.value(model.x[i,j])==1.0:
            res_dic[i][j]=p[i,j]
        #res_dic = {(i,j):pyo.value(model.x[i,j])}
        
res_df = pd.DataFrame(res_dic)
res_df = res_df.T
res_df = res_df.replace(np.nan,0)

#print(res_df)  ## This is the print out of the result
print("objective z_1 is ", pyo.value(model.z_1), "(optimized)")
print("objective z_2 is ", pyo.value(model.z_2), "(not optimized)")
optimal_z_1=pyo.value(model.z_1)


#Constraint on Z_2 for hierarchical method
eps_z_1=0.50
def hierarchical_z_1(model):
    return model.z_1 >= (1-eps_z_1)*optimal_z_1  # For PCS it should be a plus and minus for preference
model.hierarchical_z_1 = pyo.Constraint(rule=hierarchical_z_1)

def obj_rule_z2(model):
    return model.z_2
model.obj = pyo.Objective(rule = obj_rule_z2,sense=pyo.minimize)

# Solve the model and report the results 
opt = pyo.SolverFactory("cbc")

results = opt.solve(model, tee=False)


#### Display the results 

res_dic = {}
cost_res_dic = {}
count = 0
assig_res = []
for i in I:
    res_dic[i]= {}
    cost_res_dic[i]= {}
    for j in j_dic[i]:
        #print(model.x[i,j], pyo.value(model.x[i,j]))
        #res_dic[i][j] = pyo.value(model.x[i,j])
        if pyo.value(model.x[i,j])==1.0:
            res_dic[i][j]=p[i,j]
            cost_res_dic[i][j]=c[i,j]
            assig_res.append([i,j])
            if p[i,j] >= 70 :
                count +=1
        #res_dic = {(i,j):pyo.value(model.x[i,j])}
        
res_df = pd.DataFrame(res_dic)
res_df = res_df.T
res_df = res_df.replace(np.nan,0)

#print(res_df)  ## This is the print out of the result
print("Preference objective value is ", pyo.value(model.z_1), "(not optimized)")
print("PCS Cost objective value is ", pyo.value(model.z_2), "(optimized subject to z_1)")




























































