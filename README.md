# Project History
Below are selected projects done by Yuning (from the most recent to the earliest).

## Quantitative Trading and Price Impact
First established the data scope, encompassing various dataframes storing trading volumes, market prices, spreads, and stock information. Fitting OW, Reduced form, and Bouchaud price impact models and pre-computed impact models, conducted fitting procedures. Analyzed the models' performance using metrics such as in-sample and out-of-sample $R^2$ values. 

Developed a backtest engine to adjust prices for impact effects and evaluate the trading strategy's impact on prices. Generated synthetic alphas based on binned sample data, considering both basic and overnight return synthetic alphas. Devised optimal trading strategy based on a heuristic target impact formula, with performance metrics including correlation between alpha and returns, expected daily PnLs, Sharpe ratios, transaction costs, and maximum daily drawdowns. 

Conducted sensitivity analysis and stress testing, exploring the effects of varying alpha strength and introducing latency. 

## Simulation of Arbitrage-Free Implied Volatility Surfaces
Addressed arbitrage constraints on call option prices and introduced a penalty function to quantify static arbitrage violations. Proposing a Weighted Monte Carlo method, which aimed to penalize arbitrage for paths generated from a baseline model and resample according to the penalization. 

Applied this method to a factor model for the implied volatility surface, acting as the baseline model. Introduced VolGAN, describing its architecture, loss functions, reweighting method, and training process, followed by presenting numerical results and comparing the performance with and without the reweighting method.

## NSS Factors Mean Reverting Trending
Provided an overview of government bonds' significance in financial markets. Detailed the methodology for constructing the OIS curve using swap rates and interpolation techniques. Explored the NSS model and discussed its relevance in analyzing yield curve shapes and zero-coupon yields. 

Examining key parameters like $\beta$ coefficients from the NSS model, investigated the impact of economic events, central bank policies, and geopolitical factors on bond valuations. Analyzed mean-reverting tendencies and trending behaviors in the $\beta$ coefficients and summarized key findings and insights from our analysis.

## Market Microstructure
Consider the issue of hedging a European derivative security in the presence of microstructure noise. Simulate market data using Black-Scholes model and model efficient price cross the tick grid.

Implement two hedging strategy in Python: portfolio is rebalanced every time that the transaction price moves or only once the transaction price has varied by more than a selected value. Assess the hedging errors and compare statistically.

## Reinforcement Learning
Implement epsilon-greedy algorithm for designed game in Python. Train it against opponents with different strategies, assess how well does the proposed algorithm fare on each of these programmatic opponents.

Use the 10-armed testbed to compare with the greedy and epsilon-greedy method using upper-confidence-bound (UCB) action selection. Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for non-stationary problems.

## Data Science for Fintech, Suptech and Regtech
Implement Bech Algorithm and Guntzer Algorithm in Python to solve the Bank Clearing Problem, maximizing the total volume cleared while maintaining a specified level of cover money deposit.

Generate queues of payments and compare statistically the performances of the two algorithms for different initial balances and sizes of queue. Performances are assessed over a large number of queues using confidence intervals.

## Iterated Prisoner Dilemma
Utilized Python to simulate Iterated Prisoner's Dilemma (IPD) tournaments to examine the evolutionary dynamics and patterns within the population, shedding light on strategy success and adaptation over time.

## Data Science
Applied dimensionality reduction techniques in Python to analyse large datasets. Leveraged Multi-Layer Perceptron and Gaussian Mixture Models (GMM) to make predictions and extract valuable insights.

Employed the K-means clustering method for data grouping and conducted graph-based analysis to reveal hidden patterns and relationships within the data.

## Consumer Credit Risk
Develop a consumer credit risk model and improve the model by data processing, stepwise variable selection, segmentation, and testing.









