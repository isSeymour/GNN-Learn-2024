# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Q-Learning
   Description :
   Author :       Lr
   date：          2023/6/24
-------------------------------------------------
   Change Activity:
                   2023/6/24:
-------------------------------------------------
"""



import numpy as np
import random



# 建立 Q 表
q = np.zeros((7, 7))
q = np.matrix(q)

# 建立 R 表

# r = np.array([[-1, -1, -1, -1, 0, -1], [-1, -1, -1, 0, -1, 100], [-1, -1, -1, 0, -1, -1], [-1, 0, 0, -1, 0, -1],
#               [0, -1, -1, 0, -1, 100], [-1, 0, -1, -1, 0, 100]])
# r = np.matrix(r)



r = np.array([[-1,-1,-1,1,-1,-1,-1],
              [-1,-1,1,-1,-1,-1,-1],
              [-1,1,-1,1,-1,1,-1],
              [1,-1,1,-1,1,-1,-1],
              [-1,-1,-1,1,-1,1,100],
              [-1,-1,1,-1,1,-1,100],
              [-1,-1,-1,-1,1,1,100]])
r = np.matrix(r)

print(random.randint(0, 2))


# 贪婪指数
gamma = 0.8

for i in range(1000):
    # 对每一个训练,随机选择一种状态
    state = random.randint(0, 6)
    while state != 6:
        # 选择r表中非负的值的动作
        r_pos_action = []
        for action in range(7):
            if r[state, action] >= 0:
                r_pos_action.append(action)
        next_state = r_pos_action[random.randint(0, len(r_pos_action) - 1)]
        q[state, next_state] = r[state, next_state] + gamma * q[next_state].max()
        state = next_state

print("q = ",q)

state = random.randint(0, 6)
print('机器人处于{}'.format(state))
count = 0
while state != 6:
    if count > 20:   # 如果尝试次数大于20次，表示失败
        print('fail')
        break
    # 选择最大的q_max
    q_max = q[state].max()

    q_max_action = []
    for action in range(7):
        if q[state, action] == q_max: # 选择可行的下一个动作
            q_max_action.append(action)
    # 随机选择一个可行的动作
    next_state = q_max_action[random.randint(0, len(q_max_action) - 1)]
    print("机器人 goes to " + str(next_state) + '.')
    state = next_state