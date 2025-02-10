import random
import pandas as pd
import numpy as np
import time, requests, json, warnings
from scipy.stats import lognorm, uniform
from scipy.stats import norm
import scipy.stats as st 
from tqdm import tqdm
from gurobipy import *
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from math import ceil
from sklearn.utils import resample
import os
import seaborn as sns
import logging
import pickle
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from functools import partial
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)


# data generation
def generate_payment_values(size, mean, std, max_value = 10 * 1937131.63, min_value = 0): 
    """
    Generate a list of payment values following a truncated log-normal distribution.
    """
    # Calculate the lower and upper cumulative distribution function (CDF) values for truncation.
    cdf_lower = lognorm.cdf(min_value, s=std, scale=mean)
    cdf_upper = lognorm.cdf(max_value, s=std, scale=mean)

    if std == 0:
        # If standard deviation is zero, return a list of mean values.
        truncated_lognorm_samples = [mean for i in range(size)]
    else:
        # Generate uniform samples between the lower and upper CDF values.
        uniform_samples = np.random.uniform(cdf_lower, cdf_upper, size)
        # Convert uniform samples to log-normal samples.
        truncated_lognorm_samples = lognorm.ppf(uniform_samples, s=std, scale=mean)
    
    return truncated_lognorm_samples


def get_lamb(desired_payments, time_period=3600):
    """
    Calculate the lambda for a Poisson process given the desired number of payments and time period.
    """
    return time_period/desired_payments


def generate_time(lamb, start_time, end_time):
    """
    Generate a list of timestamps for payments using Poisson process.
    """
    l_times = list()
    current_time = start_time
    while current_time < end_time:
        # Generate the time delta using an exponential distribution.
        time_delta = np.round(np.random.exponential(scale=lamb, size=1),0)[0]
        current_time += timedelta(seconds = time_delta)
        if current_time < end_time:
            l_times.append(current_time)
    return l_times


def generate_datetime_fixed(lamb, start_time, end_time):
    """
    Generate a list of fixed interval timestamps for payments.
    """
    intervals_per_hour = int(3600//lamb)
    l_times = list()
    interval_duration = 60 // intervals_per_hour
    current_time = start_time
    while current_time < end_time:
        for _ in range(intervals_per_hour):
            l_times.append(current_time)
            current_time += timedelta(minutes=interval_duration)
    return l_times


def apply_fx_rate(row, rate_dict, decimal=2):
    """
    Apply foreign exchange rate to convert value to USD.
    """
    currency = row['currency']
    value = row['value']
    return round(value / rate_dict[currency+"/USD"], decimal)


def convert_value(row, rate_dict, decimal=2):
    """
    Convert value to USD or GBP based on the currency.
    """
    currency = row['currency']
    value = row['value']
    if currency=="GBP":
        return round(value * rate_dict[currency+"/USD"], decimal)
    else:
        return round(value * 1/rate_dict["GBP/"+currency], decimal)


def get_receiver_choices(row, receiver_dict):
    """
    Select a receiver based on predefined probabilities.
    """
    rng = np.random.default_rng()
    receivers = receiver_dict[row['currency']][row['sender']][0]
    probs = receiver_dict[row['currency']][row['sender']][1]
    receiver = rng.choice(receivers, 1, replace=True, p=probs)[0]
    return receiver


def verify_payments(df, rate_dict, decimal=2, verbose=False):
    df_leg_1 = df[df['leg'] == 1].copy()
    df_leg_2 = df[df['leg'] == 2].copy()
    
    df_leg_1.reset_index(drop=True, inplace=True)
    df_leg_2.reset_index(drop=True, inplace=True)
    
    discrepancies = []

    for idx, row in df_leg_1.iterrows():
        value_leg_1 = row['value']
        currency_leg_1 = row['currency']
        value_leg_2 = df_leg_2.loc[idx, 'value']
        currency_leg_2 = df_leg_2.loc[idx, 'currency']
        
        if currency_leg_1 == "GBP":
            expected_value_leg_2 = round(value_leg_1 * rate_dict["GBP/USD"], decimal)
        elif currency_leg_1 == "USD":
            expected_value_leg_2 = round(value_leg_1 / rate_dict["GBP/USD"], decimal)

        if abs(expected_value_leg_2 - value_leg_2) > 10**(-decimal):
            discrepancies.append((idx, row, expected_value_leg_2, value_leg_2))
    
    if discrepancies:
        print(f"Discrepancies found in {len(discrepancies)} rows:")
        for discrepancy in discrepancies:
            print(discrepancy)
    else:
        if verbose:
            print("All payments are verified and correct.")
        
    return discrepancies


def generate_daily_payments(num_banks, start_time, end_time,
                          lambda_dict,
                          currency_dict,
                          values_dict,
                          rate_dict,
                          receiver_dict, decimal=2, verbose=False, fix_time=False):
    
    l_dfs = list()

    previous_id = 0
    
    for i in range(num_banks):
        seed = int(os.getpid() + time.time_ns())%(2**32 - 1)
        np.random.seed(seed)
        # generate time
        l_time = generate_time(lambda_dict[i], start_time, end_time)
        if fix_time:
            l_time = generate_datetime_fixed(lambda_dict[i], start_time, end_time)

        # generate ids
        l_id = np.arange(previous_id, previous_id+len(l_time))
        previous_id += len(l_time)
        
        ## 1st leg
        # generate currency
        rng = np.random.default_rng()
        l_currency_leg_1 = rng.choice(currency_dict[i][0], len(l_time), replace=True, p=currency_dict[i][1])
        l_currency_leg_2 = np.where(l_currency_leg_1 == currency_dict[0][0][0], currency_dict[0][0][1], currency_dict[0][0][0])
        if fix_time:
            l_currency_leg_1 = ['USD' for _ in range(len(l_time))]
            l_currency_leg_2 = np.where(l_currency_leg_1 == currency_dict[0][0][0], currency_dict[0][0][1], currency_dict[0][0][0])

        # generate sender
        l_sender_leg_1 = [i for x in range(len(l_time))]

        # generate receiver
        tmp = pd.DataFrame(
            {'sender': l_sender_leg_1,
             'currency': l_currency_leg_1}
        )
        tmp['receiver'] = tmp.apply(lambda row: get_receiver_choices(row, receiver_dict), axis=1)
        l_receiver_leg_1 = tmp['receiver'].tolist()

        # generate value
        l_value_leg_1 = generate_payment_values(len(l_time), values_dict[i][0], values_dict[i][1], values_dict[i][2]) # assume USD value follow lognormal
        if decimal == 2:
            l_value_leg_1 = np.round(l_value_leg_1, decimals=decimal)
        else:
            l_value_leg_1 = np.round(l_value_leg_1, decimals=decimal-1)
        tmp = pd.DataFrame(
            {'currency': l_currency_leg_1, 
             'value': l_value_leg_1}
        )
        tmp['value_updated'] = tmp.apply(lambda row: apply_fx_rate(row, rate_dict, decimal=decimal), axis=1)
        l_value_leg_1 = tmp['value_updated'].tolist()

        # generate leg
        l_leg_1 = [1 for x in range(len(l_time))]
        
        ## 2nd leg
        # get sender and receiver
        l_sender_leg_2 = l_receiver_leg_1
        l_receiver_leg_2 = l_sender_leg_1
        
        # convert value
        tmp = pd.DataFrame(
            {'currency': l_currency_leg_1, 
             'value': l_value_leg_1}
        )
        tmp['value_leg_2'] = tmp.apply(lambda row: convert_value(row, rate_dict, decimal=decimal), axis=1)
        l_value_leg_2 = tmp['value_leg_2'].tolist()

        # generate leg
        l_leg_2 = [2 for x in range(len(l_time))]

        ## make df for participant i
        df_leg_1_i = pd.DataFrame(
            {'id': l_id,
             'time': l_time,
             'sender': l_sender_leg_1, 
             'receiver': l_receiver_leg_1, 
             'currency': l_currency_leg_1, 
             'value': l_value_leg_1,
             'leg': l_leg_1}
        )

        df_leg_2_i = pd.DataFrame(
            {'id': l_id,
             'time': l_time,
             'sender': l_sender_leg_2, 
             'receiver': l_receiver_leg_2, 
             'currency': l_currency_leg_2, 
             'value': l_value_leg_2,
             'leg': l_leg_2}
        )

        df_leg_1_i['time'] = pd.to_datetime(df_leg_1_i['time'], errors='coerce')
        df_leg_2_i['time'] = pd.to_datetime(df_leg_2_i['time'], errors='coerce')

        l_dfs.append(df_leg_1_i)
        l_dfs.append(df_leg_2_i)
    
    # merge dfs
    df = pd.concat(l_dfs, axis=0, ignore_index=True)
    df.sort_values(by=['time', 'id'], inplace=True, ignore_index=True)
    verify_payments(df, rate_dict, decimal=decimal, verbose=verbose)

    return df


# net requirement
def net_requirement(df, decimal=2):
    """
    Function return net liquidity requirement for each participant as dictionary.
    """
    # Calculate the total value sent by each sender.
    sent_values = df.groupby('sender')['value'].sum().reset_index()
    sent_values.columns = ['participant', 'total_sent']
    
    # Calculate the total value received by each receiver.
    received_values = df.groupby('receiver')['value'].sum().reset_index()
    received_values.columns = ['participant', 'total_received']
    
    # Merge the sent and received values on the participant column.
    net_values = pd.merge(sent_values, received_values, on='participant', how='outer').fillna(0)
    
    # Calculate the net requirement for each participant: total sent minus the total received, clipped at a minimum of 0.
    net_values['net_requirement'] = (net_values['total_sent'] - net_values['total_received']).clip(lower=0)
    
    # Convert the net requirement values to a dictionary with participants as keys.
    result = net_values.set_index('participant')['net_requirement'].to_dict()
    
    return result


# lnncp
def balance_history(df, lamb = 1):
    """
    Function return how balances evolves in a given day for each participant.
    """
    # Get a unique list of all participants (both senders and receivers).
    participants = np.unique([df.sender, df.receiver])
    balances = {}
    
    for p in participants:
        # Filter the dataframe to include only the payments involving the current participant.
        df_payments = df.loc[(df.sender == p) | (df.receiver == p)]
                
        if df_payments.shape == 0:
            # If there are no payments involving the participant, set their balance to 0 at the start of the day.
            balances[p] = df.time.iloc.replace(hour = 0, minute = 0, second = 0), 0, 0
            continue

        # Calculate the cumulative sum of the payment values, subtracting for sent payments and adding for received payments.
        to_cumulate = df_payments.apply(lambda x: x.value * (-1) if x.sender == p else x.value, axis=1)
        liquidity_series = to_cumulate.cumsum().set_axis(df_payments.time)
        
        # Store the liquidity series in the balances dictionary.
        balances[p] = liquidity_series

    return balances


def lnncp_function(df, decimal=2):
    """
    Function return lnncp for each participant as dictionary.
    """
    # Get a unique list of all participants (both senders and receivers)
    participants = np.unique([df.sender, df.receiver])
    lnncp_dict = {}
    
    for p in participants:
        # Filter the dataframe to include only the payments involving the current participant
        df_payments = df.loc[(df.sender == p)|(df.receiver == p)]
                
        if df_payments.shape[0] == 0:
            # If there are no payments involving the participant, set their balance to 0 at the start of the day
            lnncp_dict[p] = df.time.iloc[0].replace(hour = 0, minute = 0, second = 0), 0, 0
            continue

        # Calculate the cumulative sum of the payment values, subtracting for sent payments and adding for received payments
        to_cumulate = df_payments.apply(lambda x: x.value*(-1) if x.sender == p else x.value, axis=1)
        liquidity_series = to_cumulate.cumsum().set_axis(df_payments.time)
        
        # get the min of the balance
        lnncp_value = liquidity_series.min()
        lnncp_value = min(lnncp_value, 0.0)
        
        # get the time at which the minimum was reached
        min_datetime = to_cumulate.cumsum().set_axis(df_payments.time).idxmin()
        
        # get the total value to be cleared
        gross_cleared = df.loc[(df.sender == p)].value.sum()
        
        lnncp_dict[p] = min_datetime, round(lnncp_value, decimal), round(gross_cleared, decimal)

    return lnncp_dict


def plot_balance(df, decimal=2):
    """
    Function to plot how balances evolves in a given day for each participant.
    """
    lnncp_df = lnncp_function(df, decimal=decimal)
    balances_history_df = balance_history(df)

    color_map = plt.get_cmap('tab10')
    colors = {participant: color_map(i) for i, participant in enumerate(balances_history_df.keys())}
    
    # Single plot for all participants
    plt.figure(figsize=(14, 8))
    for participant, series in balances_history_df.items():
        plt.plot(series.index, series.values, label=f'Participant {participant}', color=colors[participant])
        plt.scatter(lnncp_df[participant][0], lnncp_df[participant][1], color=colors[participant], zorder=5)
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title('Balance Update for Each Participant')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Individual subplots for each participant
    num_participants = len(balances_history_df)
    fig, axs = plt.subplots(nrows=(num_participants + 1) // 2, ncols=2, figsize=(14, 8), sharex=True, sharey=True)
    axs = axs.flatten()
    
    for idx, (participant, ax) in enumerate(zip(balances_history_df.keys(), axs)):
        series = balances_history_df[participant]
        ax.plot(series.index, series.values, label=f'Participant {participant}', color=colors[participant])
        lnncp_time, lnncp_value = lnncp_df[participant][:2]
        ax.scatter(lnncp_time, lnncp_value, color=colors[participant], zorder=5, label='LNNCP')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_title(f'Participant {participant}')
        ax.legend()
        ax.grid(True)
    
    # Hide any unused subplots
    for ax in axs[num_participants:]:
        ax.axis('off')
    
    fig.suptitle('Balance Update for Each Participant', y=1.02)
    plt.tight_layout()
    plt.show()


# MILP
def MILP_multi(sim, day, cycle, df_A_, df_B_, balance_A_, balance_B_, current_time, log_path, model_path, file_counter, rel_tol = 0.01, 
               max_value = True, min_delay = False, priority = "A", save = False, IntFeasTol = 1e-5, TimeLimit = 180, decimal = 2, threads=4):
    """
    MILP solver function for optimizing payment processes.
    
    Parameters:
    - sim: Simulation identifier.
    - day: Current day in simulation.
    - cycle: Current cycle in simulation.
    - df_A_: DataFrame for payments of currency A.
    - df_B_: DataFrame for payments of currency B.
    - balance_A_: Initial balances for participants in currency A.
    - balance_B_: Initial balances for participants in currency B.
    - current_time: Current time in the simulation for delay calculations.
    - log_path: Path for saving logs.
    - model_path: Path for saving the Gurobi model.
    - file_counter: Counter for saving data.
    - rel_tol: Relative tolerance for the optimization.
    - max_value: Boolean to determine if maximizing payment values is a goal.
    - min_delay: Boolean to determine if minimizing payment delay is a goal.
    - priority: Determines which currency to optimize first ("A" or "B").
    - save: Boolean to indicate whether to save the model.
    - IntFeasTol: Tolerance for integer feasibility.
    - TimeLimit: Maximum time allowed for optimization.
    - decimal: Decimal precision for rounding balances.
    - threads: Number of threads for optimization.
    
    Returns:
    - Optimized DataFrames for both currencies, updated balances, value and percentage of settled payments, execution time, solution status, and number of threads used.
    """
    
    # copy the queue and the initial balance 
    df_A = df_A_.copy()
    df_B = df_B_.copy()
    balance_A = balance_A_.copy()
    balance_B = balance_B_.copy()

    # calculate total values of payments for further analysis
    A_total = df_A['value'].sum()
    B_total = df_B['value'].sum()
    
    # start timer for performance measurement
    start_time = time.time()
    
    # set id as index
    df_A.set_index(['id'],inplace=True)
    df_B.set_index(['id'],inplace=True)

    # identify participants based on balances
    participants = list(balance_A.keys())
    
    # paths for saving data
    rds_base = os.getenv('RDS')
    data_dir = fr'{rds_base}/home/{sim}/neg_balance_data'
    infeasible_dir = fr'{rds_base}/home/{sim}/infeasible_sol_data'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(infeasible_dir, exist_ok=True)
         
    # create a Gurobi environment and model
    with Env(empty=True) as env:
        env.setParam(GRB.Param.LogToConsole, 0)
        env.setParam(GRB.Param.LogFile,log_path)
        env.start()
        
        with Model(env=env) as m:
            # set the model parameters
            m.ModelSense = GRB.MAXIMIZE
            m.setParam(GRB.Param.LogToConsole, 0)
            m.setParam(GRB.Param.OutputFlag, 0)
            m.setParam(GRB.Param.IntFeasTol, IntFeasTol)
            m.setParam(GRB.Param.Threads, threads)

            # define binary decision variables for payments in A and B
            payment_ID = df_A.index
            x_A = m.addVars(payment_ID, name='x_A', vtype=GRB.BINARY)
            x_B = m.addVars(payment_ID, name='x_B', vtype=GRB.BINARY)
            
            # add linking constraint: linked payments need to be settled simultaneously
            for i in payment_ID:
                m.addConstr(x_A[i] == x_B[i], name=f'link_{i}')

            # multi-objective functions
            # Goal 1: maximize value
            if max_value:
                payment_value_B = {key:df_B.loc[key].value for key in df_B.index}
                payment_value_A = {key:df_A.loc[key].value for key in df_A.index}
                if priority == "A":
                    m.setObjectiveN(sum(x_A[i]*payment_value_A[i] for i in payment_ID), index = 0, priority = 4, reltol = rel_tol, name = "value_A")
                    m.setObjectiveN(sum(x_B[i]*payment_value_B[i] for i in payment_ID), index = 1, priority = 3, reltol = rel_tol, name = "value_B")
                elif priority == "B":
                    m.setObjectiveN(sum(x_B[i]*payment_value_B[i] for i in payment_ID), index = 0, priority = 4, reltol = rel_tol, name = "value_B")
                    m.setObjectiveN(sum(x_A[i]*payment_value_A[i] for i in payment_ID), index = 1, priority = 3, reltol = rel_tol, name = "value_A")
                else:
                    raise ValueError("Priority should be choosen within 'A' or 'B'.")

            # Goal 2: minimum delay
            if min_delay:
                if priority == "A":
                    m.setObjectiveN(sum(x_A[i]*(pd.to_datetime(current_time) - pd.to_datetime(df_A.loc[i]['time'])).total_seconds() for i in df_A.index), index=2, priority=2, name="delay_A")
                    m.setObjectiveN(sum(x_B[i]*(pd.to_datetime(current_time) - pd.to_datetime(df_B.loc[i]['time'])).total_seconds() for i in df_B.index), index=3, priority=1, name="delay_B")
                elif priority == "B":
                    m.setObjectiveN(sum(x_B[i]*(pd.to_datetime(current_time) - pd.to_datetime(df_B.loc[i]['time'])).total_seconds() for i in df_B.index), index=2, priority=2, name="delay_B")
                    m.setObjectiveN(sum(x_A[i]*(pd.to_datetime(current_time) - pd.to_datetime(df_A.loc[i]['time'])).total_seconds() for i in df_A.index), index=3, priority=1, name="delay_A")
                else:
                    raise ValueError("Priority should be choosen within 'A' or 'B'.")

            # set constraint
            # find all participants
            participants = sorted(list(set(df_A['sender']) | set(df_A['receiver'])))
            # number of payments send/received in each currency for each participant
            as_sender_payment_A = {key: list(df_A[df_A['sender'] == key].index) for key in participants}
            as_receiver_payment_A = {key: list(df_A[df_A['receiver'] == key].index) for key in participants}
            as_sender_payment_B = {key: list(df_B[df_B['sender'] == key].index) for key in participants}
            as_receiver_payment_B = {key: list(df_B[df_B['receiver'] == key].index) for key in participants}
            # initialize dictionaries for values of each payment
            directed_payment_value_A = dict()
            directed_payment_value_B = dict()
            # map payment values for each participant
            for p in participants:
                for i in as_sender_payment_A[p]:
                    directed_payment_value_A[p,i] = df_A.loc[i]['value']  # participant p as sender in currency A (money out as positive)
                for i in as_receiver_payment_A[p]:
                    directed_payment_value_A[p,i] = -df_A.loc[i]['value'] # participant p as receiver in currency A (money in as negative)
                for i in as_sender_payment_B[p]:
                    directed_payment_value_B[p,i] = df_B.loc[i]['value']  # participant p as sender in currency B (money out as positive)
                for i in as_receiver_payment_B[p]:
                    directed_payment_value_B[p,i] = -df_B.loc[i]['value'] # participant p as receiver in currency B (money in as negative)
            participant_payment_A = {p: as_sender_payment_A[p] + as_receiver_payment_A[p] for p in participants} # each participant's involved-payment
            participant_payment_B = {p: as_sender_payment_B[p] + as_receiver_payment_B[p] for p in participants} # each participant's involved-payment
            # balance is required to be non-negative
            for i in participants:
                m.addConstr(sum(directed_payment_value_A[i, j] * x_A[j] for j in participant_payment_A[i]) <= balance_A[i])
                m.addConstr(sum(directed_payment_value_B[i, j] * x_B[j] for j in participant_payment_B[i]) <= balance_B[i])

            # set and record parameters for the solver
            if save:
                m.setParam(GRB.Param.LogFile,log_path)
            m.setParam(GRB.Param.TimeLimit, TimeLimit)
            num_threads = m.getParamInfo('Threads')[2]
            
            m.optimize()

            # end timer after optimization for performance measurement
            end_time = time.time()

            # save the model if required
            if save:
                m.write(model_path)

            # get the optimal dataframe
            df_optimal_A = pd.DataFrame(columns=['sender', 'receiver', 'value'])
            df_optimal_B = pd.DataFrame(columns=['sender', 'receiver', 'value'])
            # check optimization status and retrieve solutions
            if m.status in [GRB.Status.OPTIMAL, GRB.Status.SUBOPTIMAL, GRB.Status.TIME_LIMIT]:
                x_A_sol = m.getAttr('x', x_A)
                x_B_sol = m.getAttr('x', x_B)
                
                # check for rounding error
                differences_A = {i: x_A_sol[i] - np.round(x_A_sol[i]) for i in payment_ID if x_A_sol[i] != np.round(x_A_sol[i])}
                differences_B = {i: x_B_sol[i] - np.round(x_B_sol[i]) for i in payment_ID if x_B_sol[i] != np.round(x_B_sol[i])}
                
                # find the dataframes of settled payments
                df_A['status'] = [np.round(x_A_sol[i]) for i in payment_ID]
                df_B['status'] = [np.round(x_B_sol[i]) for i in payment_ID]
                df_optimal_A = df_A[df_A['status']==1].copy().reset_index().drop('status', axis=1)
                df_optimal_B = df_B[df_B['status']==1].copy().reset_index().drop('status', axis=1)

            # record data separtately if the solution is infeasible
            else:
                logging.info(f"Day{day} Cycle{cycle} Infeasible model: data saved as counter {file_counter}.")
                sol_status = "infeasible"
                df_A_.to_csv(os.path.join(infeasible_dir, f'hourly_A_infeasible_{file_counter}.csv'), index=False)
                df_B_.to_csv(os.path.join(infeasible_dir, f'hourly_B_infeasible_{file_counter}.csv'), index=False)
                with open(os.path.join(infeasible_dir, f'cycle_balance_A_infeasible_{file_counter}.json'), 'w') as file:
                    json.dump(balance_A_, file, indent=4)
                with open(os.path.join(infeasible_dir, f'cycle_balance_B_infeasible_{file_counter}.json'), 'w') as file:
                    json.dump(balance_B_, file, indent=4)
                return df_optimal_A, df_optimal_B, balance_B, balance_A, end_time - start_time, 0, A_total, 0, 0, B_total, 0, sol_status, num_threads
            
            # record solution status
            if m.status == GRB.Status.OPTIMAL:
                sol_status = "optimal"
            elif m.status == GRB.Status.SUBOPTIMAL:
                sol_status = "suboptimal"
            elif m.status == GRB.Status.TIME_LIMIT:
                logging.info(f"Day{day} Cycle{cycle}: Time limit is reached!")
                sol_status = "time_limit"

    # update the balance
    for i in participants:
        balance_A[i] -= sum(directed_payment_value_A[i, j] * df_A.loc[j, 'status'] for j in participant_payment_A[i])
        balance_B[i] -= sum(directed_payment_value_B[i, j] * df_B.loc[j, 'status'] for j in participant_payment_B[i])
        balance_A[i] = round(balance_A[i], decimal)
        balance_B[i] = round(balance_B[i], decimal)

    # find if there is any negative balances led by rounding error
    if any(balance_A[key] < 0 or balance_B[key] < 0 for key in balance_A.keys()):
        if differences_A or differences_B:
            logging.info(f"Rounding error: data saved as counter {file_counter}.")
            df_A_.to_csv(os.path.join(data_dir, f'hourly_A_roundingerror_{file_counter}.csv'), index=False)
            df_B_.to_csv(os.path.join(data_dir, f'hourly_B_roundingerror_{file_counter}.csv'), index=False)
            with open(os.path.join(data_dir, f'cycle_balance_A_roundingerror_{file_counter}.json'), 'w') as file:
                json.dump(balance_A_, file, indent=4)
            with open(os.path.join(data_dir, f'cycle_balance_B_roundingerror_{file_counter}.json'), 'w') as file:
                json.dump(balance_B_, file, indent=4)
            A_cleared = df_optimal_A['value'].sum()
            B_cleared = df_optimal_B['value'].sum()
            sol_status = 'NA'
            
            return df_optimal_A, df_optimal_B, balance_A, balance_B, end_time - start_time, A_cleared, A_total, A_cleared/A_total, B_cleared, B_total, B_cleared/B_total, sol_status, num_threads
        
    A_cleared = df_optimal_A['value'].sum()
    B_cleared = df_optimal_B['value'].sum()
    
    return df_optimal_A, df_optimal_B, balance_A, balance_B, end_time - start_time, A_cleared, A_total, A_cleared/A_total, B_cleared, B_total, B_cleared/B_total, sol_status, num_threads


# Daily processor
def daily_processor(sim, day, df_A_, df_B_, log_path, model_path, num_banks, attempts, start_time=None, end_time=None, frequency='1H', alternate = True, verbose = False, cycle_percentage = 100, initial_percentage = 100, save = True, lnncp = False, decimal=2, threads=4):
    """
    Function processes daily settlement which settles using MILP at the end of each cycle.
    
    Parameters:
    - sim: Simulation identifier.
    - day: Current day in simulation.
    - df_A_: DataFrame for payments of currency A.
    - df_B_: DataFrame for payments of currency B.
    - log_path: Path for saving logs.
    - model_path: Path for saving the Gurobi model.
    - num_banks: Number of banks in the market (int), affecting the transaction dynamics and processing logic.
    - attempts: List of dictionaries containing model parameters for handling infeasible solutions during optimization. Each dictionary may specify parameters such as `IntFeasTol`, `cycle_balance_percentage`, and `time_limit`.
    - start_time: Start time (datetime or None) for processing transactions; defaults to 8:00 AM on the first transaction day if None.
    - end_time: End time (datetime or None) for processing transactions; defaults to 6:00 PM on the first transaction day if None.
    - frequency: Frequency (str) of processing cycles; default is '1H' (one hour), which indicates how often transactions are processed.
    - alternate: Boolean indicating whether to alternate processing priority between banks (default is True).
    - verbose: Boolean indicating whether to print detailed logs during processing (default is False).
    - cycle_percentage: Percentage (int) of available balance to use for processing in each cycle (default is 100).
    - initial_percentage: Percentage (int) of the initial balance to use at the start of processing (default is 100).
    - save: Boolean indicating whether to save intermediate results (default is True).
    - lnncp: Boolean indicating whether to use the LNNCP method for initial balance calculations (default is False).
    - decimal: Number of decimal places (int) to round financial figures to (default is 2).
    - threads: Number of threads (int) to use for parallel processing during optimization (default is 4).

    Returns:
    - df_A: Updated DataFrame for payments of currency A with settlement times and delays.
    - df_B: Updated DataFrame for payments of currency B with settlement times and delays.
    - new_balance_A: Final balances for each participant in currency A.
    - new_balance_B: Final balances for each participant in currency B.
    - rejected_count_A: Number of payments rejected for currency A.
    - rejected_count_B: Number of payments rejected for currency B.
    - rejected_value_A: Total value of payments rejected for currency A.
    - rejected_value_B: Total value of payments rejected for currency B.
    - intfeastol_list: List of tolerance values used during the optimization process.
    - reach_timelimit_list: List indicating whether the time limit was reached for each cycle.
    - cycle_percentage_list: List of cycle balance percentages used in each cycle.
    - duration_list: List of durations taken for each cycle (for each cycle there is a list which contains time taken for each attempt).
    - settlement_duration_list: List of settlement durations for completed cycles (including all attempts).
    - participants_cycle_liquidity_needed_list_A: Dictionary tracking liquidity needs per participant for currency A across cycles.
    - participants_cycle_liquidity_needed_list_B: Dictionary tracking liquidity needs per participant for currency B across cycles.
    - cycle_liquidity_needed_list_A: List of total liquidity needed for currency A per cycle.
    - cycle_liquidity_needed_list_B: List of total liquidity needed for currency B per cycle.
    - threads_list: List of the number of threads used in each cycle.
    """

    # initialization of time bounds if not provided
    if start_time is None:
        start_time = pd.to_datetime(df_A_.iloc[0]['time']).replace(hour=8, minute=0, second=0, microsecond=0)
    if end_time is None:
        end_time = pd.to_datetime(df_A_.iloc[0]['time']).replace(hour=18, minute=0, second=0, microsecond=0)

    # Creating copies of input DataFrames
    df_A = df_A_.copy()
    df_B = df_B_.copy()

    # Initialize lists and dictionaries for tracking various metrics
    intfeastol_list = []
    reach_timelimit_list = []
    cycle_percentage_list = []
    duration_list = []
    settlement_duration_list = []
    participants_cycle_liquidity_needed_list_A = {}
    participants_cycle_liquidity_needed_list_B = {}
    cycle_liquidity_needed_list_A = []
    cycle_liquidity_needed_list_B = []
    threads_list = []

    # Set up time variables and calculation of cycles per day
    current_time = start_time
    unsettled_A = pd.DataFrame(columns=df_A.columns)
    unsettled_B = pd.DataFrame(columns=df_B.columns)
    priority = "A"

    total_seconds = (end_time - start_time).total_seconds()
    cycles_per_day = total_seconds // pd.to_timedelta(frequency).total_seconds()
    cycle_percentage /= 100
    cycle_count = 0
    
    # handle initial liquidity based on lnncp flag: if not lnncp, use net requirement
    initial_percentage /= 100
    if lnncp:
        initial_A_lnncp = lnncp_function(df_A, decimal=decimal)
        initial_B_lnncp = lnncp_function(df_B, decimal=decimal)
        main_balance_A = {k: -v[1] * initial_percentage for k, v in initial_A_lnncp.items()}
        main_balance_B = {k: -v[1] * initial_percentage for k, v in initial_B_lnncp.items()}
    else:
        initial_A_net = net_requirement(df_A, decimal=decimal)
        initial_B_net = net_requirement(df_B, decimal=decimal)
        main_balance_A = {k: round(v * initial_percentage, decimal) for k, v in initial_A_net.items()}
        main_balance_B = {k: round(v * initial_percentage, decimal) for k, v in initial_B_net.items()}
    if verbose:
        print(f'Start of day balance A is {main_balance_A}')
        print(f'Start of day balance B is {main_balance_B}')
        
    # Initialize main balances for all banks
    for i in range(num_banks):
        if i not in main_balance_A:
            main_balance_A[i] = 0
        if i not in main_balance_B:
            main_balance_B[i] = 0
            
    # Create directories for saving results
    rds_base = os.getenv('RDS')
    data_dir = fr'{rds_base}/home/{sim}/neg_balance_data'
    nothelping_dir = fr'{rds_base}/home/{sim}/100net_not_working'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(nothelping_dir, exist_ok=True)
            
    # Processing loop for each time cycle
    while current_time < end_time:
        cycle_count += 1
        attempt_duration_list = []
        
        # Extract payments within the cycle
        current_time += pd.to_timedelta(frequency)
        hourly_A = df_A[(df_A['time'] >= current_time - pd.to_timedelta(frequency)) & (df_A['time'] < current_time)]
        hourly_B = df_B[(df_B['time'] >= current_time - pd.to_timedelta(frequency)) & (df_B['time'] < current_time)]

        if verbose == True:
            print(f"\nCycle {cycle_count}, current time is {current_time}, with priority: {priority}")

        # Add unsettled payments from the previous hour
        hourly_A = pd.concat([unsettled_A, hourly_A]).reset_index(drop=True)
        hourly_B = pd.concat([unsettled_B, hourly_B]).reset_index(drop=True)
        
        # Calculate net requirements for the cycle
        cycle_A_net = net_requirement(hourly_A, decimal=decimal)
        cycle_B_net = net_requirement(hourly_B, decimal=decimal)
        
        # Adjust balances based on the cycle count
        if cycle_count == cycles_per_day:
            cycle_balance_A_90 = main_balance_A
            cycle_balance_B_90 = main_balance_B
        else:
            cycle_balance_A_90 = {k: round(v * cycle_percentage, decimal) for k, v in cycle_A_net.items()}
            cycle_balance_B_90 = {k: round(v * cycle_percentage, decimal) for k, v in cycle_B_net.items()}
            
        # Ensure cycle balances do not exceed main balances
        for key in cycle_A_net:
            if main_balance_A[key] < cycle_balance_A_90[key]:
                cycle_balance_A_90[key] = main_balance_A[key]
            if main_balance_B[key] < cycle_balance_B_90[key]:
                cycle_balance_B_90[key] = main_balance_B[key]

        if not hourly_A.empty:
            # Start timing for settlement
            start_settle = datetime.fromtimestamp(time.time())
            cycle_log_path = fr'{log_path}_cycle{cycle_count}.txt'
            cycle_model_path = fr'{model_path}_cycle{cycle_count}.lp'
            file_counter = int(len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])/4)
            cycle_balance_A = cycle_balance_A_90
            cycle_balance_B = cycle_balance_B_90
            
            # Initialize cycle balances
            for i in range(num_banks):
                if i not in cycle_balance_A:
                    cycle_balance_A[i] = 0
                if i not in cycle_balance_B:
                    cycle_balance_B[i] = 0
                    
            cycle_balance_percentage = cycle_percentage
            IntFeasTol = 1e-5
            time_limit = 90
            # Call optimization function for settlements
            df_optimal_A, df_optimal_B, new_balance_A, new_balance_B, duration, A_cleared, A_total, A_ratio, B_cleared, B_total, B_ratio, sol_status, num_threads = MILP_multi(sim, day, cycle_count, hourly_A, hourly_B, cycle_balance_A, cycle_balance_B, current_time, cycle_log_path, cycle_model_path, file_counter, priority=priority, save=save, IntFeasTol=IntFeasTol, TimeLimit=time_limit, decimal=decimal, threads=threads)
            attempt_duration_list.append(duration)
            
            # Retry logic for non-optimal solutions
            for attempt in attempts:
                # rounding error happens
                if sol_status == 'NA':
                    logging.info(f"Try IntFeasTol {attempt['IntFeasTol']} and cycle balance percentage {attempt['cycle_balance_percentage']}.")
                    if cycle_count == cycles_per_day:
                        attempt_cycle_balance_A = {k: v for k, v in cycle_A_net.items()}
                        attempt_cycle_balance_B = {k: v for k, v in cycle_B_net.items()}
                    else:
                        attempt_cycle_balance_A = {k: round(v * attempt['cycle_balance_percentage']/100, decimal) for k, v in cycle_A_net.items()}
                        attempt_cycle_balance_B = {k: round(v * attempt['cycle_balance_percentage']/100, decimal) for k, v in cycle_B_net.items()}
                                 
                    # Adjust cycle balances if they exceed main balances
                    for key in cycle_A_net:
                        if main_balance_A[key] < attempt_cycle_balance_A[key]:
                            attempt_cycle_balance_A[key] = main_balance_A[key]
                        if main_balance_B[key] < attempt_cycle_balance_B[key]:
                            attempt_cycle_balance_B[key] = main_balance_B[key]
                    cycle_balance_A = attempt_cycle_balance_A
                    cycle_balance_B = attempt_cycle_balance_B
                    IntFeasTol = attempt["IntFeasTol"]
                    cycle_balance_percentage = attempt["cycle_balance_percentage"]/100
                    time_limit = attempt["time_limit"]
                    for i in range(num_banks):
                        if i not in cycle_balance_A:
                            cycle_balance_A[i] = 0
                        if i not in cycle_balance_B:
                            cycle_balance_B[i] = 0

                    # Call the optimization function again with adjusted parameters
                    df_optimal_A, df_optimal_B, new_balance_A, new_balance_B, duration, A_cleared, A_total, A_ratio, B_cleared, B_total, B_ratio, sol_status, num_threads = MILP_multi(
                        sim, day, cycle_count, hourly_A, hourly_B, cycle_balance_A, cycle_balance_B, current_time, cycle_log_path, cycle_model_path, file_counter, priority=priority, save=False, IntFeasTol=IntFeasTol, TimeLimit=time_limit, decimal=decimal, threads=threads
                    )
                    attempt_duration_list.append(duration)
                    if sol_status != 'NA':
                        break
            
            # If still not optimal, log and return
            if sol_status == 'NA':
                file_counter_nothelping = int(len([name for name in os.listdir(nothelping_dir) if os.path.isfile(os.path.join(nothelping_dir, name))])/4)
                df_A_.to_csv(os.path.join(nothelping_dir, f'hourly_A_nothelping_{file_counter_nothelping}.csv'), index=False)
                df_B_.to_csv(os.path.join(nothelping_dir, f'hourly_A_nothelping_{file_counter_nothelping}.csv'), index=False)
                return df_A, df_B, new_balance_A, new_balance_B, 1, 1, 1, 1, [], [], [], [], [], [], {}, {}, [], []
            
            if verbose == True:
                if 'settlement' in hourly_A.columns:
                    hourly_A = hourly_A.drop(columns=['settlement'])
                if 'settlement' in hourly_B.columns:
                    hourly_B = hourly_B.drop(columns=['settlement'])
                print(f"Cycle starting at {current_time - pd.to_timedelta(frequency)}")
                print(f"A payments within interval:\n{hourly_A}")
                print(f"B payments within interval:\n{hourly_B}")
                print(f"Cycle Balance A before settlement: {cycle_balance_A}")
                print(f"Cycle Balance B before settlement: {cycle_balance_B}")
                print(f"Main Balance A before settlement: {main_balance_A}")
                print(f"Main Balance B before settlement: {main_balance_B}")
            
            end_settle = datetime.fromtimestamp(time.time())
            
            cycle_liquidity_needed_A = 0
            cycle_liquidity_needed_B = 0
            for i in range(0, num_banks, 1):
                if i not in participants_cycle_liquidity_needed_list_A:
                    participants_cycle_liquidity_needed_list_A[i] = []
                if i not in participants_cycle_liquidity_needed_list_B:
                    participants_cycle_liquidity_needed_list_B[i] = []
                participants_cycle_liquidity_needed_A = -min(new_balance_A[i] - cycle_balance_A[i],0)
                participants_cycle_liquidity_needed_list_A[i].append(float(participants_cycle_liquidity_needed_A))
                cycle_liquidity_needed_A += participants_cycle_liquidity_needed_A
                participants_cycle_liquidity_needed_B = -min(new_balance_B[i] - cycle_balance_B[i],0)
                participants_cycle_liquidity_needed_list_B[i].append(float(participants_cycle_liquidity_needed_B))
                cycle_liquidity_needed_B += participants_cycle_liquidity_needed_B
            cycle_liquidity_needed_list_A.append(float(round(cycle_liquidity_needed_A, decimal)))
            cycle_liquidity_needed_list_B.append(float(round(cycle_liquidity_needed_B, decimal)))

            # Update main balances
            for key in main_balance_A:
                new_balance_value_A = new_balance_A.get(key, 0)
                cycle_balance_value_A = cycle_balance_A.get(key, 0)
                new_balance_value_B = new_balance_B.get(key, 0)
                cycle_balance_value_B = cycle_balance_B.get(key, 0)
                main_balance_A[key] += new_balance_value_A - cycle_balance_value_A
                main_balance_B[key] += new_balance_value_B - cycle_balance_value_B

            # Identify unsettled payments
            A_diff = hourly_A.merge(df_optimal_A, how='left', indicator=True)
            unsettled_A = A_diff[A_diff['_merge'] == 'left_only'].drop(columns=['_merge'])
            B_diff = hourly_B.merge(df_optimal_B, how='left', indicator=True)
            unsettled_B = B_diff[B_diff['_merge'] == 'left_only'].drop(columns=['_merge'])

            # Update settlement times for settled payments
            settled_ids_A = df_optimal_A['id'].tolist()
            settled_ids_B = df_optimal_B['id'].tolist()
            settlement_duration = end_settle - start_settle
            settlement_duration_rounded_up = settlement_duration.total_seconds()
            settlement_duration_list.append(settlement_duration_rounded_up)
            df_A.loc[df_A['id'].isin(settled_ids_A), 'settlement'] = current_time + timedelta(seconds=settlement_duration_rounded_up)
            df_B.loc[df_B['id'].isin(settled_ids_B), 'settlement'] = current_time + timedelta(seconds=settlement_duration_rounded_up)
            df_A.loc[df_A['id'].isin(settled_ids_A), 'reach_time_limit'] = 1 if sol_status == "time_limit" else 0
            df_B.loc[df_B['id'].isin(settled_ids_B), 'reach_time_limit'] = 1 if sol_status == "time_limit" else 0
            df_A.loc[df_A['id'].isin(settled_ids_A), 'IntFeasTol'] = IntFeasTol
            df_B.loc[df_B['id'].isin(settled_ids_B), 'IntFeasTol'] = IntFeasTol
            df_A.loc[df_A['id'].isin(settled_ids_A), 'cycle_balance_percentage'] = cycle_balance_percentage
            df_B.loc[df_B['id'].isin(settled_ids_B), 'cycle_balance_percentage'] = cycle_balance_percentage
            df_A.loc[df_A['id'].isin(settled_ids_A), 'time_limit'] = time_limit
            df_B.loc[df_B['id'].isin(settled_ids_B), 'time_limit'] = time_limit
            intfeastol_list.append(IntFeasTol)
            reach_timelimit_list.append(1 if sol_status == "time_limit" else 0)
            cycle_percentage_list.append(cycle_balance_percentage)
            duration_list.append(attempt_duration_list)
            threads_list.append(num_threads)

            if verbose:
                print(f"Optimal A payments:\n{df_optimal_A}")
                print(f"Optimal B payments:\n{df_optimal_B}")
                print(f"Rejected A payments:\n{unsettled_A}")
                print(f"Rejected B payments:\n{unsettled_B}")
                print(f"Cycle Balance A after settlement: {new_balance_A}")
                print(f"Cycle Balance B after settlement: {new_balance_B}")
                print(f"Main Balance A after settlement: {main_balance_A}")
                print(f"Main Balance B after settlement: {main_balance_B}")
                
        # Alternating priorities
        if alternate == True:
            priority = "A" if priority == "B" else "B"
    
    # Label remaining unsettled payments as rejected
    df_A.loc[df_A['settlement'].isnull(), 'settlement'] = 'Rejected'
    df_B.loc[df_B['settlement'].isnull(), 'settlement'] = 'Rejected'
    df_A.loc[df_A['settlement'] != 'Rejected', 'settlement'] = pd.to_datetime(df_A.loc[df_A['settlement'] != 'Rejected', 'settlement']).dt.round('S')
    df_B.loc[df_B['settlement'] != 'Rejected', 'settlement'] = pd.to_datetime(df_B.loc[df_B['settlement'] != 'Rejected', 'settlement']).dt.round('S')
    df_A.loc[df_A['settlement'] != 'Rejected', 'delay'] = (pd.to_datetime(df_A['settlement'], format='%Y-%m-%d %H:%M:%S', errors='coerce') - df_A['time']).dt.total_seconds()
    df_B.loc[df_B['settlement'] != 'Rejected', 'delay'] = (pd.to_datetime(df_B['settlement'], format='%Y-%m-%d %H:%M:%S', errors='coerce') - df_B['time']).dt.total_seconds()

    # Rejected numbers and values
    rejected_count_A = df_A[df_A['settlement'] == 'Rejected'].shape[0]
    rejected_count_B = df_B[df_B['settlement'] == 'Rejected'].shape[0]
    rejected_value_A = round(df_A[df_A['settlement'] == 'Rejected'].value.sum(), decimal)
    rejected_value_B = round(df_B[df_B['settlement'] == 'Rejected'].value.sum(), decimal)
                                 
    if rejected_count_A > 0 or rejected_count_B > 0:
        print(f'Day{day} got rejected payment.')
    
    return df_A, df_B, new_balance_A, new_balance_B, rejected_count_A, rejected_count_B, rejected_value_A, rejected_value_B, intfeastol_list, reach_timelimit_list, cycle_percentage_list, duration_list, settlement_duration_list, participants_cycle_liquidity_needed_list_A, participants_cycle_liquidity_needed_list_B, cycle_liquidity_needed_list_A, cycle_liquidity_needed_list_B, threads_list


def setup_logging(log_file_path):
    logging.basicConfig(
        filename=log_file_path,   # Log file path
        level=logging.INFO,       # Log level (INFO level will capture print statements)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d %H:%M:%S'
    )
