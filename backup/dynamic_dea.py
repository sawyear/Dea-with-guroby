import pandas as pd
import numpy as np
import gurobipy
from scipy import optimize as op
from scipy.linalg import eigh as largest_eigh
from gurobipy import quicksum
from scipy.stats import mannwhitneyu
import functools
class d_dea(object):
    def __init__(self, input_variable, desirable_output, undesirable_output, carry_over, dmu, data, weight_dmu=[], weight_year=[]):
        self.input_variable = pd.concat([data['year'], data[input_variable]], axis=1)
        self.desirable_output = pd.concat([data['year'], data[desirable_output]], axis=1)
        self.undesirable_output = pd.concat([data['year'], data[undesirable_output]], axis=1)
        self.carry_over = pd.concat([data['year'], data[carry_over]], axis=1)
        self.data, self.years = data, data['year'].unique()
        self.DMUs = self.data['dmu'].unique()
        #mannwhitneyu(a,b): Mann-Whitney-U检验是一种非参数统计方法，用于比较两个独立样本中的数据是否来自相同的总体。
        if weight_dmu == []:
            self.weight_dmu = np.ones(len(self.DMUs))/len(self.DMUs)
        if weight_year == []:
            self.weight_year = np.ones(len(self.years))/len(self.years)
        #生成五个字典：DMU，X投入，Y期望产出,Z非期望产出,C_V滞留投入
        #self.DMUs, self.X, self.Y, self.Z, self.C_V = gurobipy.multidict({DMU: [data[input_variable].loc[DMU].tolist(), data[desirable_output].loc[DMU].tolist(), data[undesirable_output].loc[DMU].tolist(), data[carry_over].loc[DMU].tolist()] for DMU in data.index})
        self.m, self.s1, self.s2, self.s3, self.s4 = len(input_variable), len(desirable_output), len(undesirable_output), len(carry_over), len(self.years)
        #结果数据框[dmu	TE	slack...]
        self.Data_res()
    #生成空白的结果数据框
    def Data_res(self):
        self.res = pd.DataFrame(columns = ['dmu','TE'] + self.years.tolist(), index = range(0,len(self.DMUs)))
        self.res['dmu'] = self.DMUs
        #for j in self.input_variable.columns[1:] + self.desirable_output.columns[1:] + self.undesirable_output.columns[1:] + self.carry_over.columns[1:]:
            #self.res[j] = np.nan
    #定义初始化结果数据的装饰器
    @staticmethod
    def reset_state(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            self.Data_res()
            return result
        return wrapper
    #动态sbm-dea
    def d_sbm(self, scale = 'c'):
        for k in range(len(self.DMUs)):
            m = gurobipy.Model()
            s_negative, s_k_positive = m.addMVar((self.m, self.s4)), m.addMVar((self.s1, self.s4)),
            s_k_negative, s_carry =  m.addMVar((self.s2, self.s4)), m.addMVar((self.s3, self.s4))
            lambdas, t =  m.addMVar((len(self.DMUs), self.s4)), m.addMVar(1)
            y_k = m.addMVar(self.s4)
            x_k = m.addMVar(self.s4)
            m.update()
            for year in range(len(self.years)):
                self.X = self.input_variable[self.input_variable['year'] == self.years[year]].iloc[:,1:].to_numpy()
                self.Y = self.desirable_output[self.desirable_output['year'] ==self.years[year]].iloc[:,1:].to_numpy()
                self.Z = self.undesirable_output[self.undesirable_output['year'] == self.years[year]].iloc[:,1:].to_numpy()
                self.C_V = self.carry_over[self.carry_over['year'] == self.years[year]].iloc[:,1:].to_numpy()
                #投入约束
                m.addConstr(self.X.T @ lambdas[:, year] == t * self.X[k] - s_negative[:, year])
                #预期产出约束
                m.addConstr(self.Y.T @ lambdas[:, year]== t * self.Y[k] + s_k_positive[:, year])
                #非预期产出约束
                m.addConstr(self.Z.T @ lambdas[:, year] == t * self.Z[k] - s_k_negative[:, year])
                #滞后效应（carry_over:-bad +good +free fix)
                m.addConstr(self.C_V.T @ lambdas[:, year] == t * self.C_V[k] - s_carry[:, year])
                #滞后效应连接
                if year + 1 < self.s4:
                    m.addConstr(self.C_V.T @ lambdas[:, year] == self.C_V.T @ lambdas[:, year + 1])
                #有效前沿类型：v~可变有效前沿，c~不变有效前沿
                if scale == 'v':
                    sbm.addConstr(lambdas[:, year].sum() == t)
                elif scale == 'c':
                    pass
                m.addConstr(y_k[year] == t + (1/(self.s1 + self.s2)) * ((s_k_positive[:, year]/self.Y[k]).sum() + (s_k_negative[:, year]/self.Z[k]).sum()))
                m.addConstr(x_k[year] == t - (1/(self.m + self.s3)) * ((s_negative[:, year]/self.X[k]).sum() + (s_carry[:, year]/self.C_V[k]).sum()))
            #charns-cooper变换
            m.addConstr(y_k @ self.weight_year == 1)
            #优化object
            m.setObjective(x_k @ self.weight_year, sense = gurobipy.GRB.MINIMIZE)
            m.setParam('OutputFlag', 0)
            m.setParam('NonConvex', 2)
            m.optimize()
            self.res.at[k, 'TE'] = m.objVal #if sbm.status == gurobipy.GRB.Status.OPTIMAL else 'N/A'
            self.res.loc[k, self.years] = x_k.X
        return self.res
    def d_n_sbm(self, scale= 'c'):
        pass
