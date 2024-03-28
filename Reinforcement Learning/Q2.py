from n_bandit import Testbed, simulate_bandit
from bandit_algo import SampleAverageMethod, ConstStepSizeMethod, UCBMethod

import matplotlib.pyplot as plt

# (e)
# Initialize the 10-bandit
bandit = Testbed(n=10, stationary=True)

# Set up methods
methods = [
    SampleAverageMethod(n=10, epsilon=0.0, opt_init=5.0),
    UCBMethod(n=10, c=2.0, epsilon=0.1),
    UCBMethod(n=10, c=2.0, epsilon=0.01)
]

# Run for 2000 episodes
labels = ['greedy $Q_0=5$', 'UCB $\epsilon=0.1$', 'UCB $\epsilon=0.01$']
colors = ['r', 'g', 'b']
ave_results = []
opt_results = []
reg_results = []

for method in methods:
    average_rewards, percent_optimal, total_regret = simulate_bandit(bandit, method, episodes=2000, steps=1000)
    ave_results.append(average_rewards)
    opt_results.append(percent_optimal)
    reg_results.append(total_regret)

# Plotting
plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 18
for i, average_rewards in enumerate(ave_results):
    plt.plot(average_rewards, label=labels[i], color=colors[i])

plt.xlabel('Plays')
plt.ylabel('Average Reward')
plt.title('Comparison of Bandit Algorithms')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 18
for i, percent_optimal in enumerate(opt_results):
    plt.plot(percent_optimal, label=labels[i], color=colors[i])

plt.xlabel('Plays')
plt.ylabel('% Optimal Action')
plt.title('Comparison of Bandit Algorithms - Percent Optimal Action')
plt.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# for i, total_regret in enumerate(reg_results):
#     plt.plot(total_regret, label=labels[i], color=colors[i])

# plt.xlabel('Plays')
# plt.ylabel('Total Regret')
# plt.title('Comparison of Bandit Algorithms - Total Regret')
# plt.legend()
# plt.show()




# (f)
# Initialize the 10-bandit
bandit = Testbed(n=10, stationary=False)

# Set up methods
methods = [
    SampleAverageMethod(n=10, epsilon=0.1, opt_init=0.0),
    ConstStepSizeMethod(n=10, epsilon=0.1, alpha=0.1)
]

# Run for 2000 episodes
labels = ['sample average', 'constant step size']
colors = ['r', 'g']
ave_results = []
opt_results = []
reg_results = []

for method in methods:
    average_rewards, percent_optimal, total_regret = simulate_bandit(bandit, method, episodes=2000, steps=10000)
    ave_results.append(average_rewards)
    opt_results.append(percent_optimal)
    reg_results.append(total_regret)

# Plotting
plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 18
for i, average_rewards in enumerate(ave_results):
    plt.plot(average_rewards, label=labels[i], color=colors[i])

plt.xlabel('Plays')
plt.ylabel('Average Reward')
plt.title('Comparison of Bandit Algorithms')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 18
for i, percent_optimal in enumerate(opt_results):
    plt.plot(percent_optimal, label=labels[i], color=colors[i])

plt.xlabel('Plays')
plt.ylabel('% Optimal Action')
plt.title('Comparison of Bandit Algorithms - Percent Optimal Action')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 18
for i, total_regret in enumerate(reg_results):
    plt.plot(total_regret, label=labels[i], color=colors[i])

plt.xlabel('Plays')
plt.ylabel('Total Regret')
plt.title('Comparison of Bandit Algorithms - Total Regret')
plt.legend()
plt.show()

