import numpy as np
import itertools
import matplotlib.pyplot as plt


# TD method

def move(string, value_func, eps):
    available = [j for j, val in enumerate(string) if val == '.']

    if np.random.rand() < eps:
        return np.random.choice(available)

    else:
        values = [value_func[string[:k] + 'X' + string[k + 1:]] for k in available]
        max_value = max(values)
        best = [k for k, v in zip(available, values) if v == max_value]
        choose = np.random.choice(best)

    return choose


# Define opponent
def random(string):
    return np.random.choice([i for i, val in enumerate(string) if val == '.'])


def win_last(string):
    for i, val in enumerate(string):
        if val == '.':
            after = string[:i] + 'O' + string[i + 1:]
            if won(after, 'O'):
                return i
    return random(string)


def win_block(string):
    for i, val in enumerate(string):
        if val == '.':
            after = string[:i] + 'X' + string[i + 1:]
            if won(after, 'X'):
                return i
    for i, val in enumerate(string):
        if val == '.':
            after = string[:i] + 'O' + string[i + 1:]
            if won(after, 'O'):
                return i
    return random(string)


# check if who won
def won(string, who):
    win_cases = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    return any(all(string[j] == who for j in case) for case in win_cases)


# Train
def train(move, opponent, episodes, alpha, eps):
    # construct states and value function
    states = {}
    characters = ['X', 'O', '.']
    comb = itertools.product(characters, repeat=9)

    comb_valid = []
    for i in comb:
        if abs(i.count('X') - i.count('O')) in [0, 1]:
            comb_valid.append(''.join(i))

    win_cases = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

    for state in comb_valid:
        X_win = any(all(state[i] == 'X' for i in case) for case in win_cases)
        O_win = any(all(state[i] == 'O' for i in case) for case in win_cases)

        if not X_win and not O_win:
            states[state] = 0.5
        elif O_win and not X_win:
            states[state] = 0
        elif X_win and not O_win:
            states[state] = 1

    n = episodes // 1000
    perform_rates = {'Win Rate': [0] * n, 'Loss Rate': [0] * n, 'Draw Rate': [0] * n}
    cum_rates = {'Win Rate': [], 'Loss Rate': [], 'Draw Rate': []}
    win_count = 0
    loss_count = 0

    # play
    for episode in range(episodes):
        s = '.........'

        while '.' in s:
            prev = s
            decision = move(s, states, eps)
            s = s[:decision] + 'X' + s[decision + 1:]
            states[prev] += alpha * (states[s] - states[prev])

            # Check if game over
            if won(s, 'X'):
                win_count += 1
                break

            if '.' not in s:
                break

            prev = s
            oppo_move = opponent(s)
            s = s[:oppo_move] + 'O' + s[oppo_move + 1:]
            states[prev] += alpha * (states[s] - states[prev])

            # Check if game over
            if won(s, 'O'):
                loss_count += 1
                break

        cum_rates['Win Rate'].append(win_count / (episode + 1))
        cum_rates['Loss Rate'].append(loss_count / (episode + 1))
        cum_rates['Draw Rate'].append((episode + 1 - loss_count - win_count) / (episode + 1))

        if (episode + 1) % 1000 == 0:
            epi = 5000
            index = (episode + 1) // 1000 - 1
            results = evaluate(states, [opponent], epi)

            perform_rates['Win Rate'][index] = results['wins'][0] / epi
            perform_rates['Draw Rate'][index] += results['draws'][0] / epi
            perform_rates['Loss Rate'][index] += results['losses'][0] / epi

    return states, perform_rates, cum_rates


def evaluate(agent, opponents, num_epi=1000):
    results = {'wins': [0] * len(opponents), 'losses': [0] * len(opponents), 'draws': [0] * len(opponents)}

    for i, opponent in enumerate(opponents):
        for j in range(num_epi):
            s = '.........'
            while '.' in s:
                decision = move(s, agent, eps=0)
                s = s[:decision] + 'X' + s[decision + 1:]

                if won(s, 'X'):
                    results['wins'][i] += 1
                    break

                if '.' not in s:
                    results['draws'][i] += 1
                    break

                oppo_move = opponent(s)
                s = s[:oppo_move] + 'O' + s[oppo_move + 1:]

                if won(s, 'O'):
                    results['losses'][i] += 1
                    break

    return results


# result
opponents = [random, win_last, win_block]

# Parameters
alpha_vals = [0.1, 0.3, 0.5]
eps_vals = [0.1, 0.2, 0.3]
episodes = 50000

# Train and evaluate agent for different params
results = []
for opponent in opponents:
    for alpha in alpha_vals:
        for eps in eps_vals:
            print(f"Training with alpha={alpha}, epsilon={eps}, opponent={opponent}")
            agent = train(move, opponent, episodes, alpha, eps)
            print("Evaluation:")
            eval_results = evaluate(agent[0], opponents)
            print(eval_results)
            results.append((alpha, eps, opponent, eval_results))

# Print final results
print("Final results:")
print(results)



# functions to plot learning curve
def plot_Cumulative_Learning_curves(*pairs):
    metrics = ['Win Rate', 'Loss Rate', 'Draw Rate']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.size'] = 18
        for rates, label in pairs:
            evaluations = rates[metric]
            x = range(1, len(evaluations) + 1)
            plt.plot(x, evaluations, label=label)
        plt.xlabel('Episodes')
        plt.ylabel(metric)
        plt.title(f'Cumulative Learning Curves: {metric}')
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_Performance_Evaluation_curves(*pairs):
    metrics = ['Win Rate', 'Loss Rate', 'Draw Rate']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.size'] = 18
        for rates, label in pairs:
            evaluations = rates[metric]
            x = [(i+1) * 1000 for i in range(len(evaluations))]
            plt.plot(x, evaluations, label=label)
        plt.xlabel('Episodes')
        plt.ylabel(metric)
        plt.title(f'Performance Evaluation Curves: {metric}')
        plt.legend()
        plt.grid(True)
        plt.show()


# Selecting proper alpha and epsilon and examine
agent1 = train(move, random, episodes, 0.5, 0.3)
agent2 = train(move, win_last, episodes, 0.5, 0.3)
agent3 = train(move, win_block, episodes, 0.5, 0.3)

plot_Cumulative_Learning_curves(
    (agent1[2], 'Agent 1'),
    (agent2[2], 'Agent 2'),
    (agent3[2], 'Agent 3')
)

plot_Performance_Evaluation_curves(
    (agent1[1], 'Agent 1'),
    (agent2[1], 'Agent 2'),
    (agent3[1], 'Agent 3')
)
