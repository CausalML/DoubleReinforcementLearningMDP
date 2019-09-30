import gym
import itertools
import numpy as np
import sys


if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.windy_gridworld import WindyGridworldEnv

from scipy.optimize import minimize, rosen, rosen_der
from scipy.optimize import Bounds

bounds = Bounds([-0.1,-0.1],[0.1,0.1])

env = CliffWalkingEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
 
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

Q_space = np.load("Q-table-cliff.npz")["xxx"]
Q_space2 = np.load("Q-table-cliff.npz")["xxx"]

prob1 = [1.0 for i in range((env.nA))]
prob1 = prob1/np.sum(prob1)


betabeta = 0.8
def sample_policy(observation,alpha=0.9):
    prob2 = alpha*Q_space[observation,:] +(1-alpha)*prob1
    return np.random.choice(env.nA,1,p=prob2)[0]
    
        
def behavior_policy(observation,beta=betabeta):
    prob2 = beta*Q_space[observation,:]+ (1-beta)*prob1
    return np.random.choice(env.nA,1,p=prob2)[0]
    
    
def target_dense(observation,alpha=0.9):
    prob2 = alpha*Q_space[observation,:]+ (1-alpha)*prob1
    return prob2

def behav_dense(observation,beta=betabeta):
    prob2 = beta*Q_space[observation,:] + (1-beta)*prob1
    return prob2

def sarsa2(env,policy, policy2,num_episodes, discount_factor=1.0,Q_space2=Q_space2, alpha= 0.6, epsilon=0.03):
   
    Q = np.copy(Q_space2)
    episode_episode = []
    
    for i_episode in range(num_episodes):

        if (i_episode + 1) % 200 == 0:

            sys.stdout.flush()
    
        state = env.reset()
        action = policy2(state)
        
        episode = []
        
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            # Pick the next action
            next_action= policy2(next_state)
            
            # TD Update
            td_target = reward + discount_factor * np.sum(Q[next_state,:]*target_dense(next_state))
            td_delta = td_target - Q[state,action]
            Q[state,action] += alpha * td_delta 
    
            if done:
                break
                
            action = next_action
            state = next_state 
        episode_episode.append(episode)
    
    return Q, episode_episode

bounds = Bounds([-0.2,-0.2],[0.2,0.2])
def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))


depth = 1
def mc_prediction(env, policy,policy2, episode_episode, Q_=1.0,num_episodes=100, discount_factor=1.0):
   

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    returns_count2 = defaultdict(float)
 
    predic_list = []
    predic_list2 = []
    predic_list3 = []
    predic_list22 = []
    predic_list4 = []
    predic_list5 = np.ones(num_episodes)
    auxiauxi = [] 
    epiepi = []
    weight_list = np.zeros([num_episodes,1000]) ### For bounded IPW
    weight_list2 = np.zeros([num_episodes,1002]) ### For bounded IPW
    weight_list3 = np.zeros([num_episodes,1002]) ### For bounded IPW
    marginal_weight = np.zeros([num_episodes,1000]) ### For bounded IPW
    marginal_weight_2 = np.zeros([num_episodes,1000]) ### For bounded IPW
    auxi_list = np.zeros([num_episodes,1000])
    marginal_auxi_list2 = np.zeros([num_episodes,1000])
    marginal_auxi_list = np.zeros([num_episodes,1000])
    marginal_auxi_list2_2 = np.zeros([num_episodes,1000])
    marginal_auxi_list_2 = np.zeros([num_episodes,1000])
    auxi_list2 = np.zeros([num_episodes,1000])
    reward_list = np.zeros([num_episodes,1000])
    state_list = np.zeros([num_episodes,1000])
    action_list = np.zeros([num_episodes,1000])
    
    count_list = np.zeros(1000) 
    episolode_longe_list = []
    

    for i_episode in range(num_episodes):
       
        if i_episode % 200 == 0:
          
            sys.stdout.flush()
        episode = episode_episode[i_episode]
     
        W = 1.0
        W_list = []
        episolode_longe_list.append(len(episode))
        
        weight_list2[i_episode,0] = 1.0
        for t in range(len(episode)):
            state, action, reward = episode[t]
            reward_list[i_episode,t] = reward
            state_list[i_episode,t] = state
            action_list[i_episode,t] = action
            
            W = W*target_dense(state)[action]/behav_dense(state)[action]*discount_factor
            probprob = 0.9*Q_space[state,:] + 0.1*prob1
            W_list.append(W)
            weight_list[i_episode,t] = W_list[t]
            weight_list2[i_episode,t+1] = W_list[t]
            weight_list3[i_episode,t] = target_dense(state)[action]/behav_dense(state)[action]
            
            count_list[t] += 1.0
            
            if t==0:
                auxi_list[i_episode,t] = W_list[t]*Q_[state,action]-np.sum(probprob*Q_[state,:])
            else:
                auxi_list[i_episode,t] = W_list[t]*Q_[state,action]-W_list[t-1]*np.sum(probprob*Q_[state,:])
          
            if t==0:
                auxi_list2[i_episode,t] = W_list[t]-1.0
            else:
                auxi_list2[i_episode,t] = W_list[t]-W_list[t-1]

    print np.max(np.array(episolode_longe_list))
    
        
    weight_list_mean = np.mean(weight_list,1)
    reward_list_mean = np.mean(reward_list,1)
    auxi_list_mean = np.mean(auxi_list,1)
    auxi_list2_mean = np.mean(auxi_list2,1)
    
    val = []    
 
    ##### IPW
    for i in range(num_episodes):
        predic_list.append(np.sum(weight_list[i,:]*reward_list[i,:]))   
    
    val.append(np.mean(predic_list))
    
    #### Marginalized-IPW 
    
    for i in range(num_episodes):
        for j in range(episolode_longe_list[i]):
            marginal_weight[i,j] = np.mean(weight_list[:,j][(state_list[:,j]==state_list[i,j]) & (action_list[:,j]==action_list[i,j])])
            if j==0:
                marginal_weight_2[i,j] = weight_list3[i,j]
            else:
                marginal_weight_2[i,j] = np.mean(weight_list[:,j-1][(state_list[:,j]==state_list[i,j])])*weight_list3[i,j]
    
    
    for i_episode in range(num_episodes):
        for t in range(episolode_longe_list[i_episode]):
            state = np.int(state_list[i_episode,t])
            action = np.int(action_list[i_episode,t])
            probprob = 0.9*Q_space[state,:] + 0.1*prob1
            if t==0:
                marginal_auxi_list[i_episode,t] = marginal_weight[i_episode,t]*Q_[state,action]-np.sum(probprob*Q_[state,:])
                marginal_auxi_list_2[i_episode,t] = marginal_weight_2[i_episode,t]*Q_[state,action]-np.sum(probprob*Q_[state,:])
                auxi_list[i_episode,t] = weight_list[i_episode,t]*Q_[state,action]-np.sum(probprob*Q_[state,:])
            else:
                marginal_auxi_list[i_episode,t] = marginal_weight[i_episode,t]*(Q_[state,action])-marginal_weight[i_episode,t-1]*np.sum(probprob*(Q_[state,:]))
                marginal_auxi_list_2[i_episode,t] = marginal_weight_2[i_episode,t]*(Q_[state,action])-marginal_weight_2[i_episode,t-1]*np.sum(probprob*(Q_[state,:]))
                auxi_list[i_episode,t] = weight_list[i_episode,t]*(Q_[state,action])-weight_list[i_episode,t-1]*np.sum(probprob*(Q_[state,:]))
          
            if t==0:
                marginal_auxi_list2[i_episode,t] = marginal_weight[i_episode,t]-1.0
                marginal_auxi_list2_2[i_episode,t] = marginal_weight_2[i_episode,t]-1.0
                auxi_list2[i_episode,t] = weight_list[i_episode,t]-1.0
            else:
                marginal_auxi_list2[i_episode,t] =  marginal_weight[i_episode,t]- marginal_weight[i_episode,t-1]
                marginal_auxi_list2_2[i_episode,t] =  marginal_weight_2[i_episode,t]- marginal_weight_2[i_episode,t-1]
                auxi_list2[i_episode,t] = weight_list[i_episode,t]-weight_list[i_episode,t-1]

    
    for i in range(num_episodes):
        predic_list2.append(np.sum(marginal_weight[i,:]*reward_list[i,:]))   
    
    ### marginal ipw2  #### Using action and state 
    val.append(np.mean(predic_list2))
        

    ### marginal ipw3#### Using only state 
    for i in range(num_episodes):
        predic_list22.append(np.sum(marginal_weight_2[i,:]*reward_list[i,:]))   
    
    val.append(np.mean(predic_list22))
   
  
    #### DR
    val.append(np.mean(predic_list)-np.mean(np.sum(auxi_list,1)))
    
    #### marginal DR 1  #### Using action and state 
    val.append(np.mean(predic_list2)-np.mean(np.sum(marginal_auxi_list,1)))
    #### marginal DR 2   #### Using only state                                     
    val.append(np.mean(predic_list22)-np.mean(np.sum(marginal_auxi_list_2,1)))
    
    


    return val
                                                  
    




is_list = []
is2_list = []
is3_list = []
wis_list = []
wis2_list = []
dm_list = []
dr_list = []
dr2_list = []
dr3_list = []
bdr_list = []
drs_list = []
drs2_list = []
drss_list = []
mdr_list = []
mdr_list2 = []

sample_size = 1000
sample_size =sample_size/2
for kkk in range(100):
    print "epoch",kkk
    #### Sample splititng 
    ### First fold 
    
    predicted_Q ,episode_episode = sarsa2(env,sample_policy,behavior_policy, sample_size)
    V_10k_1  = mc_prediction(env,sample_policy,behavior_policy, episode_episode, predicted_Q,num_episodes=sample_size)
    
    ### Second fold 
    predicted_Q ,episode_episode = sarsa2(env,sample_policy,behavior_policy, sample_size)
    V_10k_2  = mc_prediction(env,sample_policy,behavior_policy, episode_episode, predicted_Q,num_episodes=sample_size)
    
    V_10k = 0.5*(np.array(V_10k_1)+np.array(V_10k_2))
    is_list.append(np.mean(V_10k[0]))
    is2_list.append(np.mean(V_10k[1]))
    is3_list.append(np.mean(V_10k[2]))
    dr_list.append(np.mean(V_10k[3]))
    dr2_list.append(np.mean(V_10k[4]))   
    dr3_list.append(np.mean(V_10k[5]))  
    probprob = 0.9*Q_space[36,:] + 0.1*prob1
    dm_list.append(np.sum(probprob*predicted_Q[36,:]))
    np.savez("2estimator_list_ipw_"+str(betabeta)+"_"+str(sample_size),a=is_list)
    np.savez("2estimator_list_ipw2_"+str(betabeta)+"_"+str(sample_size), a=is3_list)
    np.savez("2estimator_list_dm_"+str(betabeta)+"_"+str(sample_size), a=dm_list)
    np.savez("2estimator_list_dr_"+str(betabeta)+"_"+str(sample_size),a=dr_list)
    np.savez("2estimator_list_dr2_"+str(betabeta)+"_"+str(sample_size),a=dr3_list)




true = -42.49
def mse(aaa):
    aaa = np.array(aaa)
    aaa = aaa[aaa>-100]
    return [np.mean((((aaa-true)*(aaa-true)))),np.sqrt(np.var((aaa-true)*(aaa-true)))]

print np.mean(is_list)
print mse(is_list)
print "wis"
print np.mean(is3_list)
print mse(is3_list)
print "dm"
print np.mean(dm_list)
print mse(dm_list)
print "dr"
print np.mean(dr_list)
print mse(dr_list)
print "dr3"
print np.mean(dr3_list)
print mse(dr3_list)