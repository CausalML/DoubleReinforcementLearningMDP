import numpy as np
import scipy as sp
import sys
import math





def sigmoid(x):
    return(1.0/(1.0+np.exp(-0.1*x)))

### Both models are good  

beta = 0.2
alpha = 0.9
rep = 1500
num = 0 
estimator_list_ipw = []
estimator_list_ipw2 = []
estimator_list_ipw3 = []
estimator_list_dm = []
estimator_list_dr = []
estimator_list_dr2 = []
estimator_list_dr3 = []
estimator_list_ipw2_ratio_mis = []
estimator_list_dr2_ratio_mis = []
estimator_list_dm_q_mis = []
estimator_list_dr_q_mis = []
estimator_list_dr2_q_mis = []


args = sys.argv
ver_ = str(args[1])
N = np.int(args[2])


from sklearn import linear_model

for iii in range(rep): 
    
    T =  30

    r_list = np.zeros([N,T])
    weight_list = np.zeros([N,T])
    s_list = np.zeros([N,T])
    a_list = np.zeros([N,T])
    w_list = np.zeros([N,T])
    w_list2 = np.zeros([N,T])
    
    def behav_policy(s,i):
        a = beta*sigmoid(s)+(beta)*np.random.uniform(0.0,1.0)
        return(np.random.binomial(1,a,1)[0])

    def eval_policy(s,i):
        a = alpha*sigmoid(s)+(1-alpha)*np.random.uniform(0.0,1.0)
        return(np.random.binomial(1,a,1)[0])
    
    def behav_policy_dens(s,a,i):
        b = beta*sigmoid(s)+(beta)*0.5
        if a==1:
            return(b)
        else:
            return(1.0-b)

    def eval_policy_dens(s,a,i):
        b = alpha*sigmoid(s)+(1-alpha)*0.5
        if a==1:
            return(b)
        else:
            return(1.0-b)
        
        
    for i in range(N):
        for j in range(T):
            if j==0:
                s = np.random.normal(0.5,0.2)
                r = 0.0
                a =0.0
                w = 1.0
            else:
                s = np.random.normal(0.02*(j%2)+s*1.0-0.3*(a-0.5),0.2)
            a = behav_policy(s,j)
            w = eval_policy_dens(s,a,j)/behav_policy_dens(s,a,j)*w
            r = np.random.normal(0.9*s+0.3*a-0.02*(j%2),0.2)
            r_list[i,j] = r
            s_list[i,j] = s 
            a_list[i,j] = a 
            w_list[i,j] = w
            w_list2[i,j]= eval_policy_dens(s,a,j)/behav_policy_dens(s,a,j)
    
    ag_list = []
    
    #### IPW estimator
    for i in range(N):
        ag_list.append(np.sum(r_list[i,]*w_list[i,]))
    estimator_list_ipw.append(np.mean(ag_list))
    
    ########num = 0 
    
    #### DM estimator 
    bbb = range(T)
    reg_list = []
    for j in bbb[::-1]:
        if j==(T-1):
            X = np.array([s_list[:,j],a_list[:,j]])
            pre_X = np.array([s_list[:,j],a_list[:,j]])
            Y = r_list[:,j]
        else:
            X = np.array([s_list[:,j],a_list[:,j]])
            aaa = []
            for k in range(N):
                aaa.append(eval_policy_dens(s_list[k,j+1],1,0))
            X0 = np.array([s_list[:,j+1],aaa])
            Y = r_list[:,j]+reg.predict(np.transpose(X0))
        reg = linear_model.LinearRegression()
        reg.fit(np.transpose(X), Y)
        ###print reg.score(np.transpose(X), Y)
        reg_list.append(reg)
    

    aaa = []
    for i in range(N):
        aaa.append(eval_policy_dens(s_list[i,0],1,0))
    X0 = np.array([s_list[:,0],aaa])   
    v0 = reg.predict(np.transpose(X0)) 
    estimator_list_dm.append(np.mean(v0))
    
    ### DR estiamtor under M_1
    dr = 0.0
    for t in range(T):
        dr = dr + np.mean(r_list[:,t]*w_list[:,t])
        #### q function
        X = np.array([s_list[:,t],a_list[:,t]])
        dr = dr - np.mean(reg_list[T-1-t].predict(np.transpose(X))*w_list[:,t])
        #### v function 
        aaa = []
        for i in range(N):
            aaa.append(eval_policy_dens(s_list[i,t],1,0))
        X0 = np.array([s_list[:,t],aaa])
        if t==0:
            dr = dr + np.mean(reg_list[T-1-t].predict(np.transpose(X0)))
        else:
            dr = dr + np.mean(reg_list[T-1-t].predict(np.transpose(X0))*w_list[:,t-1])
    
    estimator_list_dr.append(dr)
    
    #### IPW estimator under M_2
    
    bbb = range(T)
    wreg_list = []
    for j in bbb[::-1]:
        X = np.array([s_list[:,j],a_list[:,j]])
        Y = w_list[:,j]
        reg = linear_model.LinearRegression()
        reg.fit(np.transpose(X),Y)
        ###print reg.score(np.transpose(X), Y)
        wreg_list.append(reg)
        
    ipw = 0.0
    for t in range(T):
        X = np.array([s_list[:,t],a_list[:,t]])
        ipw = ipw + np.mean(wreg_list[T-1-t].predict(np.transpose(X))*r_list[:,t])
    estimator_list_ipw2.append(ipw)
    
    ### DR estiamtor under M_2
    dr2 = 0.0
    for t in range(T):
        X = np.array([s_list[:,t],a_list[:,t]])
        dr2 = dr2 + np.mean(wreg_list[T-1-t].predict(np.transpose(X))*r_list[:,t])
        #### q function
        dr2 = dr2 - np.mean(reg_list[T-1-t].predict(np.transpose(X))*wreg_list[T-1-t].predict(np.transpose(X)))
        #### v function 
        aaa = []
        for i in range(N):
            aaa.append(eval_policy_dens(s_list[i,t],1,0))
        X0 = np.array([s_list[:,t],aaa])
        if t==0:
            dr2 = dr2 + np.mean(reg_list[T-t-1].predict(np.transpose(X0))) 
        else:
            X_ = np.array([s_list[:,t-1],a_list[:,t-1]])
            dr2 = dr2 + np.mean(reg_list[T-1-t].predict(np.transpose(X0))*wreg_list[T-t].predict(np.transpose(X_)))
    estimator_list_dr2.append(dr2)
        
     #### Ratio-mis specified 
    
    num = 2
    
    bbb = range(T)
    wreg_list_mis = []
    for j in bbb[::-1]:
        X = np.array([s_list[:,j]*s_list[:,j],a_list[:,j]])
        Y = w_list[:,j]
        reg = linear_model.LinearRegression()
        reg.fit(np.transpose(X),Y)
        ###print reg.score(np.transpose(X), Y)
        wreg_list_mis.append(reg)
        
    ipw = 0.0
    for t in range(T):
        X = np.array([s_list[:,t]*s_list[:,t],a_list[:,t]])
        ipw = ipw + np.mean(wreg_list_mis[T-1-t].predict(np.transpose(X))*r_list[:,t])
    estimator_list_ipw2_ratio_mis.append(ipw)
   
    dr2 = 0.0
    for t in range(T):
        X_w = np.array([s_list[:,t]*s_list[:,t],a_list[:,t]])
        X_r = np.array([s_list[:,t],a_list[:,t]])
        dr2 = dr2 + np.mean(wreg_list_mis[T-1-t].predict(np.transpose(X_w))*r_list[:,t])
        #### q function
        dr2 = dr2 - np.mean(reg_list[T-1-t].predict(np.transpose(X_r))*wreg_list_mis[T-1-t].predict(np.transpose(X_w)))
        #### v function 
        aaa = []
        for i in range(N):
            aaa.append(eval_policy_dens(s_list[i,t],1,0))
        X0 = np.array([s_list[:,t],aaa])
        if t==0:
            dr2 = dr2 + np.mean(reg_list[T-t-1].predict(np.transpose(X0))) 
        else:
            X_ = np.array([s_list[:,t-1]*s_list[:,t-1],a_list[:,t-1]])
            dr2 = dr2 + np.mean(reg_list[T-1-t].predict(np.transpose(X0))*wreg_list_mis[T-t].predict(np.transpose(X_)))
    estimator_list_dr2_ratio_mis.append(dr2)
    
    
    ### q-misspcified 
    
    
    #### DM estimator 
    bbb = range(T)
    reg_list_mis = []
    for j in bbb[::-1]:
        if j==(T-1):
            X = np.array([s_list[:,j]*s_list[:,j],a_list[:,j]])
            pre_X = np.array([s_list[:,j],a_list[:,j]])
            Y = r_list[:,j]
        else:
            X = np.array([s_list[:,j]*s_list[:,j],a_list[:,j]])
            aaa = []
            for k in range(N):
                aaa.append(eval_policy_dens(s_list[k,j+1],1,0))
            X0 = np.array([s_list[:,j+1]*s_list[:,j+1],aaa])
            Y = r_list[:,j]+reg.predict(np.transpose(X0))
        reg = linear_model.LinearRegression()
        reg.fit(np.transpose(X), Y)
        ###print reg.score(np.transpose(X), Y)
        reg_list_mis.append(reg)
    

    aaa = []
    for i in range(N):
        aaa.append(eval_policy_dens(s_list[i,0],1,0))
    X0 = np.array([s_list[:,0],aaa])   
    v0 = reg.predict(np.transpose(X0)) 
    estimator_list_dm_q_mis.append(np.mean(v0))
    
    ### DR estiamtor under M_1
    dr = 0.0
    for t in range(T):
        dr = dr + np.mean(r_list[:,t]*w_list[:,t])
        #### q function
        X = np.array([s_list[:,t]*s_list[:,t],a_list[:,t]])
        dr = dr - np.mean(reg_list_mis[T-1-t].predict(np.transpose(X))*w_list[:,t])
        #### v function 
        aaa = []
        for i in range(N):
            aaa.append(eval_policy_dens(s_list[i,t],1,0))
        X0 = np.array([s_list[:,t]*s_list[:,t],aaa])
        if t==0:
            dr = dr + np.mean(reg_list_mis[T-1-t].predict(np.transpose(X0)))
        else:
            dr = dr + np.mean(reg_list_mis[T-1-t].predict(np.transpose(X0))*w_list[:,t-1])
    
    estimator_list_dr_q_mis.append(dr)
    
    print iii
    

    ### DR estiamtor under M_2
    dr2 = 0.0
    for t in range(T):
        X_w = np.array([s_list[:,t],a_list[:,t]])
        X_r = np.array([s_list[:,t]*s_list[:,t],a_list[:,t]])
        dr2 = dr2 + np.mean(wreg_list[T-1-t].predict(np.transpose(X_w))*r_list[:,t])
        #### q function
        dr2 = dr2 - np.mean(reg_list_mis[T-1-t].predict(np.transpose(X_r))*wreg_list[T-1-t].predict(np.transpose(X_w)))
        #### v function 
        aaa = []
        for i in range(N):
            aaa.append(eval_policy_dens(s_list[i,t],1,0))
        X0 = np.array([s_list[:,t]*s_list[:,t],aaa])
        if t==0:
            dr2 = dr2 + np.mean(reg_list_mis[T-t-1].predict(np.transpose(X0))) 
        else:
            X_ = np.array([s_list[:,t-1],a_list[:,t-1]])
            dr2 = dr2 + np.mean(reg_list_mis[T-1-t].predict(np.transpose(X0))*wreg_list[T-t].predict(np.transpose(X_)))
    estimator_list_dr2_q_mis.append(dr2)

    
    np.savez("estimator_list_ipw_%d"+ver_+"_"+str(N),a=estimator_list_ipw)
    np.savez("estimator_list_dr_%d"+ver_+"_"+str(N), a=estimator_list_dr)
    np.savez("estimator_list_dm_%d"+ver_+"_"+str(N), a=estimator_list_dm)
    np.savez("estimator_list_ipw2_%d"+ver_+"_"+str(N),a=estimator_list_ipw2)
    np.savez("estimator_list_dr2_%d"+ver_+"_"+str(N),a=estimator_list_dr2)
    np.savez("estimator_list_ipw3_%d"+ver_+"_"+str(N),a=estimator_list_ipw3)
    np.savez("estimator_list_dr3_%d"+ver_+"_"+str(N),a=estimator_list_dr3)
    np.savez("estimator_list_ipw2_ratio_mis_%d"+ver_+"_"+str(N),a=estimator_list_ipw2_ratio_mis)
    np.savez("estimator_list_dr2_ratio_mis_%d"+ver_+"_"+str(N),a=estimator_list_dr2_ratio_mis)
    np.savez("estimator_list_dm_q_mis_%d"+ver_+"_"+str(N),a=estimator_list_dm_q_mis)
    np.savez("estimator_list_dr_q_mis_%d"+ver_+"_"+str(N),a=estimator_list_dr_q_mis)
    np.savez("estimator_list_dr2_q_mis_%d"+ver_+"_"+str(N),a=estimator_list_dr2_q_mis)
    
        

