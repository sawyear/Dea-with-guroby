import pandas as pd
import numpy as np
import copy
import gurobipy
from scipy import optimize as op
from scipy.linalg import eigh as largest_eigh
from dea import Dea
from gurobipy import quicksum
class gurobi_Dea(object):
    """用于求解sbm、ebm和network模型
    Parameters:
    -----------
    input_variable:
        投入[x1,x2,x3,...]
    desirable_output:
        期望产出[y1,y2,y3,...]
    undesirable_output:
        非期望产出[w1,w2,w3,...]
    dmu:
        决策单元['dmu']
    data:
        主数据
    method：
    -----------
    self.sbm:
        求解基于SBM的DEA
    self.add:
        求解基于add的DEA
    self.ebm:
        求解基于EBM的DEA
    self.n_sbm:
        求解基于网络SBM的DEA
    self.n_ebm:
        求解基于网络EBM的DEA
    Return:
    ------
    res : DataFrame
        结果数据框[dmu	TE	slack...]
    """
    def __init__(self, input_variable, desirable_output, undesirable_output, dmu, data):
        self.input_variable, self.desirable_output, self.undesirable_output, self.data = input_variable, desirable_output, undesirable_output, data
        #生成四个字典：DMB，投入，期望产出和非期望产出
        self.DMUs, self.X, self.Y, self.Z = gurobipy.multidict({DMU: [data[input_variable].loc[DMU].tolist(), data[desirable_output].loc[DMU].tolist(), data[undesirable_output].loc[DMU].tolist()] for DMU in data.index})
        self.m, self.s1, self.s2 = len(input_variable), len(desirable_output), len(undesirable_output)
        #结果数据框[dmu	TE	slack...]
        self.res = pd.DataFrame(columns = ['dmu','TE'], index = data.index)
        self.res['dmu'] = data[dmu]
        for j in input_variable + desirable_output + undesirable_output:
            self.res[j] = np.nan

    def s_corr(self, a, b):
        C = np.log(np.array(b)/np.array(a))
        if max(C) == min(C):
            return 1
        else:
            D = np.average(np.abs(C - np.average(C)))/(max(C)-min(C))
            return 1 - 2 * D

    def affinity_matrix(self, direction = "input", rem = np.array([])):
        '''
        用来计算affinity_matrix
        '''
        re = self.sbm('v')
        if direction == "input" :
            re_matrix = self.data.loc[:, self.input_variable] - re.loc[:, self.input_variable]
            if self.m > 1:
                S = np.eye(self.m)
                for i in range(self.m):
                    for j in range(self.m):
                        S[i,j] = self.s_corr(re_matrix.iloc[:,i], re_matrix.iloc[:,j])
                pho, w_x = largest_eigh(S, eigvals=(self.m-1,self.m-1))
                epsilon = (self.m - pho)/(self.m - 1)
                w_ = w_x/(w_x.sum())
            else:
                epsilon, w_ = 0, [1]
        elif direction == "output":
            re_matrix = re.loc[:, self.desirable_output + self.undesirable_output] + self.data.loc[:,self.desirable_output + self.undesirable_output]
            if self.s1 + self.s2 > 1:
                s0 = self.s1 + self.s2
                S = np.eye(s0)
                for i in range(s0):
                    for j in range(s0):
                        S[i,j] = self.s_corr(re_matrix.iloc[:,i], re_matrix.iloc[:,j])
                pho, w_x = largest_eigh(S, eigvals=(s0-1,s0-1))
                epsilon = (s0 - pho)/(s0 - 1)
                w_ = w_x/(w_x.sum())
            else:
                epsilon, w_ = 0, [1]
        elif direction == "desirable_output":
            re_matrix = re.loc[:, self.desirable_output] + self.data.loc[:, self.desirable_output]
            if self.s1 > 1:
                S = np.eye(self.s1)
                for i in range(self.s1):
                    for j in range(self.s1):
                        S[i,j] = self.s_corr(re_matrix.iloc[:,i], re_matrix.iloc[:,j])
                pho, w_x = largest_eigh(S, eigvals=(self.s1-1,self.s1-1))
                epsilon = (self.s1 - pho)/(self.s1 - 1)
                w_ = w_x/(w_x.sum())
            else:
                epsilon, w_ = 0, [1]
        elif direction == "undesirable_output":
            re_matrix = self.data.loc[:, self.undesirable_output] - re.loc[:, self.undesirable_output]
            if self.s2 > 1:
                S = np.eye(self.s2)
                for i in range(self.s2):
                    for j in range(self.s2):
                        S[i,j] = self.s_corr(re_matrix.iloc[:,i], re_matrix.iloc[:,j])
                pho, w_x = largest_eigh(S, eigvals=(self.s2-1,self.s2-1))
                epsilon = (self.s2 - pho)/(self.s2 - 1)
                w_ = w_x/(w_x.sum())
            else:
                epsilon, w_ = 0, [1]
        else:
            if rem.shape[1] > 1:
                S = np.eye(rem.shape[1])
                for i in range(rem.shape[1]):
                    for j in range(rem.shape[1]):
                        S[i,j] = self.s_corr(rem.iloc[:,i], rem.iloc[:,j])
                pho, w_x = largest_eigh(S, eigvals=(rem.shape[1]-1, rem.shape[1]-1))
                epsilon = (rem.shape[1] - pho)/(rem.shape[1] - 1)
                w_ = w_x/(w_x.sum())
            else:
                epsilon, w_ = 0, [1]
        return epsilon, w_

    def sbm(self, scale = 'c'):
        for k in self.DMUs:
            sbm = gurobipy.Model()
            s_negative, lambdas, s_k_positive, s_k_negative, t = sbm.addVars(self.m), sbm.addVars(self.DMUs), sbm.addVars(self.s1), sbm.addVars(self.s2), sbm.addVar()
            sbm.update()
            sbm.setObjective(t - (1/self.m) * quicksum(s_negative[i] / self.X[k][i] for i in range(self.m)), sense = gurobipy.GRB.MINIMIZE)
            sbm.addConstrs(quicksum(self.X[i][j] * lambdas[i] for i in self.DMUs) == t * self.X[k][j] - s_negative[j] for j in range(self.m))
            sbm.addConstrs(quicksum(self.Y[i][j] * lambdas[i] for i in self.DMUs) == t * self.Y[k][j] + s_k_positive[j] for j in range(self.s1))
            sbm.addConstrs(quicksum(self.Z[i][j] * lambdas[i] for i in self.DMUs) == t * self.Z[k][j] - s_k_negative[j] for j in range(self.s2))
            sbm.addConstr(t + (1/(self.s1 + self.s2)) * (quicksum(s_k_positive[i] / self.Y[k][i] for i in range(self.s1)) + quicksum(s_k_negative[i] / self.Z[k][i] for i in range(self.s2))) == 1)
            if scale == 'v':
                sbm.addConstr(quicksum(lambdas[i] for i in self.DMUs) == t)
            elif scale == 'c':
                pass
            sbm.setParam('OutputFlag', 0)
            sbm.setParam('NonConvex', 2)
            sbm.optimize()
            self.res.at[k, 'TE'] = sbm.objVal #if sbm.status == gurobipy.GRB.Status.OPTIMAL else 'N/A'
            for i in range(self.m):
                self.res.loc[k,self.input_variable[i]] = s_negative[i].X/(t.X or 1)
            for i in range(self.s1):
                self.res.loc[k,self.desirable_output[i]] = s_k_positive[i].X/(t.X or 1)
            for i in range(self.s2):
                self.res.loc[k,self.undesirable_output[i]] = s_k_negative[i].X/(t.X or 1)
        return self.res

    def sbm_manual(self, scale, input_variable, desirable_output, undesirable_output, dmu, data, method = 'revised simplex'):
        res = pd.DataFrame(columns = ['dmu','TE'], index = data.index)
        res['dmu'] = data[dmu]
        ## lambda有dmu个数个，S有变量个数个
        dmu_counts = data.shape[0]
        ## 投入个数
        m = len(input_variable)
        ## 期望产出个数
        s1 = len(desirable_output)
        ## 非期望产出个数
        s2 = len(undesirable_output)
        total = dmu_counts + m + s1 + s2
        cols = input_variable+desirable_output+ undesirable_output
        newcols = []
        for j in cols:
            newcols.append(j)
            res[j] = np.nan
        for i in range(dmu_counts):
            ## 优化目标
            c = [0] * dmu_counts + [1] +  list(-1 / (m * data.loc[i, input_variable])) + [0] * (s1 + s2)
            ## 约束条件
            A_eq = [[0] * dmu_counts + [1] + [0] * m  + list(1/((s1 + s2) * data.loc[i, desirable_output])) + list(1/((s1 + s2) * data.loc[i, undesirable_output]))]
            ## 约束条件（1）
            for j1 in range(m):
                list1 = [0] * m
                list1[j1] = 1
                eq1 = list(data[input_variable[j1]]) + [-data.loc[i ,input_variable[j1]]] + list1 + [0] * (s1 + s2)
                A_eq.append(eq1)
            ## 约束条件（2）
            for j2 in range(s1):
                list2 = [0] * s1
                list2[j2] = -1
                eq2 = list(data[desirable_output[j2]]) + [-data.loc[i, desirable_output[j2]]] + [0] * m + list2 + [0] * s2
                A_eq.append(eq2)
            ## 约束条件（3）
            for j3 in range(s2):
                list3 = [0] * s2
                list3[j3] = 1
                eq3 = list(data[undesirable_output[j3]]) + [-data.loc[i, undesirable_output[j3]]] + [0] * (m + s1) + list3
                A_eq.append(eq3)
            if scale == 'c':
                b_eq = [1] + [0] * (m + s1 + s2)
                bounds = [(0, None)] * (total + 1)
            elif scale == 'v':
                eq4 = list([1] * dmu_counts + [-1] + [0] * (m + s1 + s2))
                A_eq.append(eq4)
                b_eq = [1] + [0] * (m + s1 + s2) + [0]
                bounds = [(0, None)] * (total + 1)
            ## 求解
            op1 = op.linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
            res.loc[i, 'TE'] = op1.fun
            #res.loc[i, newcols] = op1.x[dmu_counts+1 :]/op1.x[dmu_counts]
            res.loc[i, newcols] = np.where(op1.x[dmu_counts] != 0, op1.x[dmu_counts+1 :]/op1.x[dmu_counts], 0)
        return res

    def add(self, scale = 'c'):
        for k in self.DMUs:
            add = gurobipy.Model()
            s_negative, lambdas, s_k_positive, s_k_negative = add.addVars(self.m), add.addVars(self.DMUs), add.addVars(self.s1), add.addVars(self.s2)
            add.update()
            #add.setObjective(quicksum(s_negative[i] / self.X[k][i] for i in range(self.m)) + quicksum(s_k_positive[i] / self.Y[k][i] for i in range(self.s1)), sense = gurobipy.GRB.MAXIMIZE)
            add.setObjectiveN(quicksum(s_negative[i] / self.X[k][i] for i in range(self.m)), index = 0, weight = 1)
            add.setObjectiveN(quicksum(s_k_positive[i] / self.Y[k][i] for i in range(self.s1)), index = 0, weight = 1)
            add.addConstrs(quicksum(self.X[i][j] * lambdas[i] for i in self.DMUs) == self.X[k][j] - s_negative[j] for j in range(self.m))
            add.addConstrs(quicksum(self.Y[i][j] * lambdas[i] for i in self.DMUs) == self.Y[k][j] + s_k_positive[j] for j in range(self.s1))
            add.addConstrs(quicksum(self.Z[i][j] * lambdas[i] for i in self.DMUs) == self.Z[k][j] - s_k_negative[j] for j in range(self.s2))
            if scale == 'v':
                add.addConstr(quicksum(lambdas[i] for i in self.DMUs) == 1)
            elif scale == 'c':
                pass
            add.setParam('OutputFlag', 0)
            add.setParam('NonConvex', 2)
            #add.setAttr('ModelSense', gurobipy.GRB.MAXIMIZE)
            add.setAttr(gurobipy.GRB.Attr.ModelSense, gurobipy.GRB.MAXIMIZE)
            add.optimize()
            self.res.at[k, 'TE'] = add.objVal #if add.status == gurobipy.GRB.Status.OPTIMAL else 'N/A'
            for i in range(self.m):
                self.res.loc[k,self.input_variable[i]] = s_negative[i].X
            for i in range(self.s1):
                self.res.loc[k,self.desirable_output[i]] = s_k_positive[i].X
            for i in range(self.s2):
                self.res.loc[k,self.undesirable_output[i]] = s_k_negative[i].X
        return self.res

    def ebm(self, scale = 'c'):
        epsilon_x, w_x = self.affinity_matrix('input')
        epsilon_y, w_y = self.affinity_matrix('desirable_output')
        epsilon_z, w_z = self.affinity_matrix('undesirable_output')
        for k in self.DMUs:
            ebm = gurobipy.Model()
            theta, eta = ebm.addVar(), ebm.addVar()
            t = ebm.addVar()
            s_negative, s_k_positive, s_k_negative, lambdas = ebm.addVars(self.m), ebm.addVars(self.s1), ebm.addVars(self.s2), ebm.addVars(self.DMUs)
            ebm.update()
            ebm.setObjective(theta - float(epsilon_x) * quicksum(float(w_x[i]) * s_negative[i] / self.X[k][i] for i in range(self.m)), sense = gurobipy.GRB.MINIMIZE)
            ebm.addConstrs(quicksum(self.X[i][j] * lambdas[i] for i in self.DMUs) == theta * self.X[k][j] - s_negative[j] for j in range(self.m))
            ebm.addConstrs(quicksum(self.Y[i][j] * lambdas[i] for i in self.DMUs) == eta * self.Y[k][j] + s_k_positive[j] for j in range(self.s1))
            ebm.addConstrs(quicksum(self.Z[i][j] * lambdas[i] for i in self.DMUs) == eta * self.Z[k][j] - s_k_negative[j] for j in range(self.s2))
            ebm.addConstr(eta + float(epsilon_y) * quicksum(float(w_y[i]) * s_k_positive[i] / self.Y[k][i] for i in range(self.s1)) + float(epsilon_z) * quicksum(float(w_z[i]) * s_k_negative[i] / self.Z[k][i] for i in range(self.s2)) == 1)
            if scale == 'v':
                ebm.addConstr(quicksum(lambdas[i] for i in self.DMUs) == 1)
            elif scale == 'c':
                pass
            ebm.setParam('OutputFlag', 0)
            ebm.setParam('NonConvex', 2)
            ebm.optimize()
            self.res.at[k, 'TE'] = ebm.objVal #if ebm.status == gurobipy.GRB.Status.OPTIMAL else 'N/A'
            for i in range(self.m):
                self.res.loc[k,self.input_variable[i]] = s_negative[i].X
            for i in range(self.s1):
                self.res.loc[k,self.desirable_output[i]] = s_k_positive[i].X
            for i in range(self.s2):
                self.res.loc[k,self.undesirable_output[i]] = s_k_negative[i].X
        return self.res

    def ebm_plus(self, scale = 'c'):
        epsilon_x, w_x = self.affinity_matrix('input')
        epsilon_y, w_y = self.affinity_matrix('desirable_output')
        epsilon_z, w_z = self.affinity_matrix('undesirable_output')
        for k in self.DMUs:
            ebm = gurobipy.Model()
            theta, eta = ebm.addVar(), ebm.addVar()
            t = ebm.addVar()
            s_negative, s_k_positive, s_k_negative, lambdas = ebm.addVars(self.m), ebm.addVars(self.s1), ebm.addVars(self.s2), ebm.addVars(self.DMUs)
            ebm.update()
            ebm.setObjective(theta * t - float(epsilon_x) * t * quicksum(float(w_x[i]) * s_negative[i] / self.X[k][i] for i in range(self.m)), sense = gurobipy.GRB.MINIMIZE)
            ebm.addConstrs(quicksum(self.X[i][j] * lambdas[i] for i in self.DMUs) == theta * self.X[k][j] - s_negative[j] for j in range(self.m))
            ebm.addConstrs(quicksum(self.Y[i][j] * lambdas[i] for i in self.DMUs) == eta * self.Y[k][j] + s_k_positive[j] for j in range(self.s1))
            ebm.addConstrs(quicksum(self.Z[i][j] * lambdas[i] for i in self.DMUs) == eta * self.Z[k][j] - s_k_negative[j] for j in range(self.s2))
            ebm.addConstr(eta * t + float(epsilon_y) * t * quicksum(float(w_y[i]) * s_k_positive[i] / self.Y[k][i] for i in range(self.s1)) + float(epsilon_z) * t * quicksum(float(w_z[i]) * s_k_negative[i] / self.Z[k][i] for i in range(self.s2)) == 1)
            if scale == 'v':
                ebm.addConstr(quicksum(lambdas[i] for i in self.DMUs) == 1)
            elif scale == 'c':
                pass
            ebm.setParam('OutputFlag', 0)
            ebm.setParam('NonConvex', 2)
            ebm.optimize()
            self.res.at[k, 'TE'] = ebm.objVal #if ebm.status == gurobipy.GRB.Status.OPTIMAL else 'N/A'
            for i in range(self.m):
                self.res.loc[k,self.input_variable[i]] = s_negative[i].X
            for i in range(self.s1):
                self.res.loc[k,self.desirable_output[i]] = s_k_positive[i].X
            for i in range(self.s2):
                self.res.loc[k,self.undesirable_output[i]] = s_k_negative[i].X
        return self.res

    def n_sbm(self, scale = 'c', x1=[], y1=[], z1=[], w12=[], x2=[], y2=[], z2=[], w23=[], x3=[], y3=[], z3=[], step=2):
        self.despoit, self.x1, self.y1, self.z1, self.w12 = gurobipy.multidict({DMU: [self.data[x1].loc[DMU].tolist(), self.data[y1].loc[DMU].tolist(), self.data[z1].loc[DMU].tolist(), self.data[w12].loc[DMU].tolist()] for DMU in self.data.index})
        self.despoit, self.x2, self.y2, self.z2, self.w23 = gurobipy.multidict({DMU: [self.data[x2].loc[DMU].tolist(), self.data[y2].loc[DMU].tolist(), self.data[z2].loc[DMU].tolist(), self.data[w23].loc[DMU].tolist()] for DMU in self.data.index})
        self.despoit, self.x3, self.y3, self.z3 = gurobipy.multidict({DMU: [self.data[x3].loc[DMU].tolist(), self.data[y3].loc[DMU].tolist(), self.data[z3].loc[DMU].tolist()] for DMU in self.data.index})
        for k in self.DMUs:
            n_sbm = gurobipy.Model()
            s_x1, s_y1, s_z1, s_x2, s_y2, s_z2 = n_sbm.addVars(len(x1)),n_sbm.addVars(len(y1)),n_sbm.addVars(len(z1)),n_sbm.addVars(len(x2)),n_sbm.addVars(len(y2)),n_sbm.addVars(len(z2))
            s_x3, s_y3, s_z3, lambdas, t = n_sbm.addVars(len(x3)),n_sbm.addVars(len(y3)),n_sbm.addVars(len(z3)), n_sbm.addVars(3, self.DMUs), n_sbm.addVar()
            n_sbm.update()
            #优化目标
            n_sbm.setObjective(
                t - 1/step * (
                    quicksum(s_x1[j]/(len(x1) * self.x1[k][j]) for j in range(len(x1)))+
                    quicksum(s_x2[j]/(len(x2) * self.x2[k][j]) for j in range(len(x2)))+
                    quicksum(s_x3[j]/(len(x3) * self.x3[k][j]) for j in range(len(x3)))
                ), sense = gurobipy.GRB.MINIMIZE
            )
            #n_sbm.setObjectiveN(t, weight = 1, index = 0)
            #n_sbm.setObjectiveN(quicksum(s_x1[j]/(len(x1) * self.x1[k][j]) for j in range(len(x1))), weight = - 1/step, index = 1)
            #n_sbm.setObjectiveN(quicksum(s_x2[j]/(len(x2) * self.x2[k][j]) for j in range(len(x2))), weight = - 1/step, index = 2)
            #n_sbm.setObjectiveN(quicksum(s_x3[j]/(len(x3) * self.x3[k][j]) for j in range(len(x3))), weight = - 1/step, index = 3)
            #约束条件(input，output)
            n_sbm.addConstrs(quicksum(self.x1[i][j] * lambdas[1,i] for i in self.DMUs) == t * self.x1[k][j] - s_x1[j] for j in range(len(x1)))
            n_sbm.addConstrs(quicksum(self.y1[i][j] * lambdas[1,i] for i in self.DMUs) == t * self.y1[k][j] + s_y1[j] for j in range(len(y1)))
            n_sbm.addConstrs(quicksum(self.z1[i][j] * lambdas[1,i] for i in self.DMUs) == t * self.z1[k][j] - s_z1[j] for j in range(len(z1)))
            n_sbm.addConstrs(quicksum(self.x2[i][j] * lambdas[2,i] for i in self.DMUs) == t * self.x2[k][j] - s_x2[j] for j in range(len(x2)))
            n_sbm.addConstrs(quicksum(self.y2[i][j] * lambdas[2,i] for i in self.DMUs) == t * self.y2[k][j] + s_y2[j] for j in range(len(y2)))
            n_sbm.addConstrs(quicksum(self.z2[i][j] * lambdas[2,i] for i in self.DMUs) == t * self.z2[k][j] - s_z2[j] for j in range(len(z2)))
            n_sbm.addConstrs(quicksum(self.x3[i][j] * lambdas[3,i] for i in self.DMUs) == t * self.x3[k][j] - s_x3[j] for j in range(len(x3)))
            n_sbm.addConstrs(quicksum(self.y3[i][j] * lambdas[3,i] for i in self.DMUs) == t * self.y3[k][j] + s_y3[j] for j in range(len(y3)))
            n_sbm.addConstrs(quicksum(self.z3[i][j] * lambdas[3,i] for i in self.DMUs) == t * self.z3[k][j] - s_z3[j] for j in range(len(z3)))
            #约束条件(link,t)
            n_sbm.addConstrs(quicksum(self.w12[i][j] * lambdas[1,i] - self.w12[i][j] * lambdas[2,i] for i in self.DMUs) == 0 for j in range(len(w12)))
            n_sbm.addConstrs(quicksum(self.w23[i][j] * lambdas[2,i] - self.w23[i][j] * lambdas[3,i] for i in self.DMUs) == 0 for j in range(len(w23)))
            n_sbm.addConstr(
                t + 1/step * (
                        quicksum(s_y1[j]/(len(y1 + z1) * self.y1[k][j]) for j in range(len(y1))) +
                        quicksum(s_z1[j]/(len(y1 + z1) * self.z1[k][j]) for j in range(len(z1))) +
                        quicksum(s_y2[j]/(len(y2 + z2) * self.y2[k][j]) for j in range(len(y2))) +
                        quicksum(s_z2[j]/(len(y2 + z2) * self.z2[k][j]) for j in range(len(z2))) +
                        quicksum(s_y3[j]/(len(y3 + z3) * self.y3[k][j]) for j in range(len(y3))) +
                        quicksum(s_z3[j]/(len(y3 + z3) * self.z3[k][j]) for j in range(len(z3)))
                )
                == 1
            )
            if scale == 'v':
                for i in range(step):
                    n_sbm.addConstr(quicksum(lambdas[i,j] for j in self.DMUs) == t)
            elif scale == 'c':
                pass
            #设置参数
            n_sbm.setParam('OutputFlag', 0)
            n_sbm.setParam('NonConvex', 2)
            n_sbm.setAttr(gurobipy.GRB.Attr.ModelSense, gurobipy.GRB.MINIMIZE)
            #优化
            n_sbm.optimize()
            self.res.at[k, 'TE'] = n_sbm.objVal #if n_sbm.status == gurobipy.GRB.Status.OPTIMAL else 'N/A' #如果不是最优就取N/A
            for i in range(len(x1)):
                self.res.loc[k,x1[i]] = s_x1[i].X/t.X
            for i in range(len(y1)):
                self.res.loc[k,y1[i]] = s_y1[i].X/t.X
            for i in range(len(z1)):
                self.res.loc[k,z1[i]] = s_z1[i].X/t.X
            for i in range(len(x2)):
                self.res.loc[k,x2[i]] = s_x2[i].X/t.X
            for i in range(len(y2)):
                self.res.loc[k,y2[i]] = s_y2[i].X/t.X
            for i in range(len(z2)):
                self.res.loc[k,z2[i]] = s_z2[i].X/t.X
            for i in range(len(x3)):
                self.res.loc[k,x3[i]] = s_x3[i].X/t.X
            for i in range(len(y3)):
                self.res.loc[k,y3[i]] = s_y3[i].X/t.X
            for i in range(len(z3)):
                self.res.loc[k,z3[i]] = s_z3[i].X/t.X
        return self.res

    def n_ebm(self, scale = 'c', x1=[], y1=[], z1=[], w12=[], x2=[], y2=[], z2=[], w23=[], x3=[], y3=[], z3=[], step=2):
        self.abandon, self.x1, self.y1, self.z1, self.w12 = gurobipy.multidict({DMU: [self.data[x1].loc[DMU].tolist(), self.data[y1].loc[DMU].tolist(), self.data[z1].loc[DMU].tolist(), self.data[w12].loc[DMU].tolist()] for DMU in self.data.index})
        self.abandon, self.x2, self.y2, self.z2, self.w23 = gurobipy.multidict({DMU: [self.data[x2].loc[DMU].tolist(), self.data[y2].loc[DMU].tolist(), self.data[z2].loc[DMU].tolist(), self.data[w23].loc[DMU].tolist()] for DMU in self.data.index})
        self.abandon, self.x3, self.y3, self.z3 = gurobipy.multidict({DMU: [self.data[x3].loc[DMU].tolist(), self.data[y3].loc[DMU].tolist(), self.data[z3].loc[DMU].tolist()] for DMU in self.data.index})
        for k in self.DMUs:
            n_ebm = gurobipy.Model()
            s_x1, s_y1, s_z1, s_x2, s_y2, s_z2 = n_ebm.addVars(len(x1)), n_ebm.addVars(len(y1)), n_ebm.addVars(len(z1)), n_ebm.addVars(len(x2)), n_ebm.addVars(len(y2)), n_ebm.addVars(len(z2))
            s_x3, s_y3, s_z3, lambdas, theta, eta = n_ebm.addVars(len(x3)),n_ebm.addVars(len(y3)),n_ebm.addVars(len(z3)), n_ebm.addVars(3, self.DMUs), n_ebm.addVars(3), n_ebm.addVars(3)
            #params = [w_x1, w_x2, w_x3, w_y1, w_y2, w_y3, w_z1, w_z2, w_z3, epsilon_x1, epsilon_x2, epsilon_x3, epsilon_y1, epsilon_y2, epsilon_y3, epsilon_z1, epsilon_z2, epsilon_z3]
            sbm_1 = self.sbm_manual(scale = 'v', input_variable = x1, desirable_output = y1 + w12, undesirable_output = z1, dmu = ['dmu'], data = self.data)
            sbm_2 = self.sbm_manual(scale = 'v', input_variable = x2 + w12, desirable_output = y2 + w23, undesirable_output = z2, dmu = ['dmu'], data = self.data)
            sbm_3 = self.sbm_manual(scale = 'v', input_variable = x3 + w23, desirable_output = y3 + w12, undesirable_output = z3, dmu = ['dmu'], data = self.data)
            epsilon_x1, w_x1 = self.affinity_matrix(direction='manual', rem=self.data.loc[:, x1] - sbm_1.loc[:, x1])
            epsilon_x2, w_x2 = self.affinity_matrix(direction='manual', rem=self.data.loc[:, x2] - sbm_2.loc[:, x2])
            epsilon_x3, w_x3 = self.affinity_matrix(direction='manual', rem=self.data.loc[:, x3] - sbm_3.loc[:, x3])
            epsilon_y1, w_y1 = self.affinity_matrix(direction='manual', rem=self.data.loc[:, y1] + sbm_1.loc[:, y1])
            epsilon_y2, w_y2 = self.affinity_matrix(direction='manual', rem=self.data.loc[:, y2] + sbm_2.loc[:, y2])
            epsilon_y3, w_y3 = self.affinity_matrix(direction='manual', rem=self.data.loc[:, y3] + sbm_3.loc[:, y3])
            epsilon_z1, w_z1 = self.affinity_matrix(direction='manual', rem=self.data.loc[:, z1] - sbm_1.loc[:, z1])
            epsilon_z2, w_z2 = self.affinity_matrix(direction='manual', rem=self.data.loc[:, z2] - sbm_2.loc[:, z2])
            epsilon_z3, w_z3 = self.affinity_matrix(direction='manual', rem=self.data.loc[:, z3] - sbm_3.loc[:, z3])
            n_ebm.update()
            #优化目标
            n_ebm.setObjective(
                theta[0]+ theta[1] + theta[2] - 1/step * (
                    quicksum(s_x1[j] * w_x1[j] * epsilon_x1/self.x1[k][j] for j in range(len(x1))) +
                    quicksum(s_x2[j] * w_x2[j] * epsilon_x2/self.x2[k][j] for j in range(len(x2))) +
                    quicksum(s_x3[j] * w_x3[j] * epsilon_x3/self.x3[k][j] for j in range(len(x3)))
                ), sense = gurobipy.GRB.MINIMIZE
            )
            #n_ebm.setObjectiveN(quicksum(theta[i] for i in range(3)), weight = 1, index = 0)
            #n_ebm.setObjectiveN(quicksum(s_x1[j] * w_x1[j] * epsilon_x1/self.x1[k][j] for j in range(len(x1))), weight = - 1/step, index = 1)
            #n_ebm.setObjectiveN(quicksum(s_x2[j] * w_x2[j] * epsilon_x2/self.x2[k][j] for j in range(len(x2))), weight = - 1/step, index = 2)
            #n_ebm.setObjectiveN(quicksum(s_x3[j] * w_x3[j] * epsilon_x3/self.x3[k][j] for j in range(len(x3))), weight = - 1/step, index = 3)
            #约束条件(input，output)
            n_ebm.addConstrs(quicksum(self.x1[i][j] * lambdas[1,i] for i in self.DMUs) == theta[0] * self.x1[k][j] - s_x1[j] for j in range(len(x1)))
            n_ebm.addConstrs(quicksum(self.y1[i][j] * lambdas[1,i] for i in self.DMUs) == eta[0] * self.y1[k][j] + s_y1[j] for j in range(len(y1)))
            n_ebm.addConstrs(quicksum(self.z1[i][j] * lambdas[1,i] for i in self.DMUs) == eta[0] * self.z1[k][j] - s_z1[j] for j in range(len(z1)))
            n_ebm.addConstrs(quicksum(self.x2[i][j] * lambdas[2,i] for i in self.DMUs) == theta[1] * self.x2[k][j] - s_x2[j] for j in range(len(x2)))
            n_ebm.addConstrs(quicksum(self.y2[i][j] * lambdas[2,i] for i in self.DMUs) == eta[1] * self.y2[k][j] + s_y2[j] for j in range(len(y2)))
            n_ebm.addConstrs(quicksum(self.z2[i][j] * lambdas[2,i] for i in self.DMUs) == eta[1] * self.z2[k][j] - s_z2[j] for j in range(len(z2)))
            n_ebm.addConstrs(quicksum(self.x3[i][j] * lambdas[3,i] for i in self.DMUs) == theta[2] * elf.x3[k][j] - s_x3[j] for j in range(len(x3)))
            n_ebm.addConstrs(quicksum(self.y3[i][j] * lambdas[3,i] for i in self.DMUs) == eta[2] * sself.y3[k][j] + s_y3[j] for j in range(len(y3)))
            n_ebm.addConstrs(quicksum(self.z3[i][j] * lambdas[3,i] for i in self.DMUs) == eta[2] * sself.z3[k][j] - s_z3[j] for j in range(len(z3)))
            #约束条件(link,t)
            n_ebm.addConstrs(quicksum(self.w12[i][j] * lambdas[1,i] - self.w12[i][j] * lambdas[2,i] for i in self.DMUs) == 0 for j in range(len(w12)))
            n_ebm.addConstrs(quicksum(self.w23[i][j] * lambdas[2,i] - self.w23[i][j] * lambdas[3,i] for i in self.DMUs) == 0 for j in range(len(w23)))
            n_ebm.addConstr(
                eta[0]+ eta[1] + eta[2] + 1/step * (
                        quicksum(s_y1[j] * w_y1[j] * epsilon_y1/self.y1[k][j] for j in range(len(y1))) +
                        quicksum(s_z1[j] * w_z1[j] * epsilon_z1/self.z1[k][j] for j in range(len(z1))) +
                        quicksum(s_y2[j] * w_y2[j] * epsilon_y2/self.y2[k][j] for j in range(len(y2))) +
                        quicksum(s_z2[j] * w_z2[j] * epsilon_z2/self.z2[k][j] for j in range(len(z2))) +
                        quicksum(s_y3[j] * w_y3[j] * epsilon_y3/self.y3[k][j] for j in range(len(y3))) +
                        quicksum(s_z3[j] * w_z3[j] * epsilon_z3/self.z3[k][j] for j in range(len(z3)))
                ) == 1
            )
            if scale == 'v':
                for i in range(step):
                    n_ebm.addConstr(quicksum(lambdas[i,j] for j in self.DMUs) == 1)
            elif scale == 'c':
                pass
            #设置参数
            n_ebm.setParam('OutputFlag', 0)
            n_ebm.setParam('NonConvex', 2)
            n_ebm.Params.LogToConsole=True # 显示求解过程
            #优化
            n_ebm.optimize()
            self.res.at[k, 'TE'] = n_ebm.objVal #if n_ebm.status == gurobipy.GRB.Status.OPTIMAL else 'N/A'
            for i in range(len(x1)):
                self.res.loc[k,x1[i]] = s_x1[i].X
            for i in range(len(y1)):
                self.res.loc[k,y1[i]] = s_y1[i].X
            for i in range(len(z1)):
                self.res.loc[k,z1[i]] = s_z1[i].X
            for i in range(len(x2)):
                self.res.loc[k,x2[i]] = s_x2[i].X
            for i in range(len(y2)):
                self.res.loc[k,y2[i]] = s_y2[i].X
            for i in range(len(z2)):
                self.res.loc[k,z2[i]] = s_z2[i].X
            for i in range(len(x3)):
                self.res.loc[k,x3[i]] = s_x3[i].X
            for i in range(len(y3)):
                self.res.loc[k,y3[i]] = s_y3[i].X
            for i in range(len(z3)):
                self.res.loc[k,z3[i]] = s_z3[i].X
        return self.res

    def d_n_sbm(self):
        pass

'''
d2 = gurobi_Dea(input_variable= ['x1_1','x1_2','x2_1'], desirable_output=['y2_1','y2_2'], undesirable_output=['w2_1','w2_2','w2_3'], dmu = ['dmu'], data = data1)
d2.sbm()
d2.add()
d2.ebm()
d2.ebm_plus()
d2.n_sbm(x1=['x1_1','x1_2'],x2=['x2_1'],y1=[],w12=['z12_1','z12_2'],y2=['y2_1','y2_2'],z2=['w2_1','w2_2','w2_3'])
d2.n_ebm(x1=['x1_1','x1_2'],x2=['x2_1'],y1=[],w12=['z12_1','z12_2'],y2=['y2_1','y2_2'],z2=['w2_1','w2_2','w2_3'])
'''
