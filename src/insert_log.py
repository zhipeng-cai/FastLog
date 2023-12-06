import pandas as pd

positions = pd.read_csv('../output/predictions/new-dataset/positions.csv')
statements = pd.read_csv('../output/predictions/new-dataset/statements_greedy_search.csv')
positions["LogStatement"] = statements["LogStatement"]
positions["Pred_Position"] = positions.apply(lambda x: x["Pred_Position"].replace("<LOG>", x["LogStatement"]), axis=1)
results = positions[["Input", "Pred_Position"]]
results.columns = ["Input", "Prediction"]
results.to_csv('../output/predictions/new-dataset/results_greedy_search.csv')