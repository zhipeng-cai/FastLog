import pandas as pd

from bleu_calculator import my_corpus_bleu
from rouge_calculator import cal_rouge


def cal_position_metrics(prediction_file, label_file):
    print("Evaluate log position...")
    correct_class = 1
    predictions = pd.read_csv(prediction_file)["Position"]
    target_labels = pd.read_csv(label_file)["Label"]
    targets = target_labels.apply(lambda x: [int(num) for num in x.split()].index(correct_class))
    data = pd.DataFrame()
    data["prediction"] = predictions
    data["target"] = targets
    data["match"] = data["prediction"] == data["target"]
    print("Accuracy: {}     Correct: {}     Total: {}".format(100 * len(data[data["match"] == True]) / len(data),
                                                              len(data[data["match"] == True]), len(data)))


def cal_level_metrics(predict_file, label_file):
    print("Evaluate log level...")
    cat_dict = {'trace': 0, 'debug': 1, 'info': 2, 'warn': 3, 'error': 4, 'fatal': 5}
    predictions = pd.read_csv(predict_file)["Level"]
    targets = pd.read_csv(label_file)["Level"]
    targets = targets.replace(cat_dict)
    data = pd.DataFrame()
    data["prediction"] = predictions
    data["target"] = targets
    data["match"] = data["prediction"] == data["target"]
    print("Accuracy: {}     Correct: {}     Total: {}".format(100 * len(data[data["match"] == True]) / len(data),
                                                              len(data[data["match"] == True]), len(data)))


def cal_message_metrics(prediction_file, label_file):
    print("Evaluate log message...")
    predictions = pd.read_csv(prediction_file)["Message"]
    predictions.fillna(" ", inplace=True)
    targets = pd.read_csv(label_file)["Message"]
    data = pd.DataFrame()
    data["prediction"] = predictions.apply(lambda x: x.replace(' ', ''))
    data["target"] = targets.apply(lambda x: x.replace(' ', ''))
    data["match"] = data["prediction"] == data["target"]
    print("Accuracy: {}     Correct: {}     Total: {}".format(100 * len(data[data["match"] == True]) / len(data),
                                                              len(data[data["match"] == True]), len(data)))

    my_corpus_bleu(predictions.tolist(), targets.tolist(), True)
    cal_rouge(predictions.tolist(), targets.tolist())


def cal_all_metrics(position_prediction_file, position_label_file, logsta_prediction_file, logsta_label_file):
    print("Evaluate all log aspects...")
    correct_class = 1
    position_predictions = pd.read_csv(position_prediction_file)["Position"]
    position_target_labels = pd.read_csv(position_label_file)["Label"]
    position_targets = position_target_labels.apply(lambda x: [int(num) for num in x.split()].index(correct_class))

    logsta_predictions = pd.read_csv(logsta_prediction_file)["LogStatement"]
    logsta_targets = pd.read_csv(logsta_label_file)["LogStatement"]
    logsta_predictions.fillna(" ", inplace=True)
    logsta_predictions = logsta_predictions.apply(lambda x: x.replace(' ', ''))
    logsta_targets = logsta_targets.apply(lambda x: x.replace(' ', ''))

    data = pd.DataFrame()
    data["position_prediction"] = position_predictions
    data["position_target"] = position_targets
    data["logsta_prediction"] = logsta_predictions
    data["logsta_target"] = logsta_targets
    data["match"] = (data["position_prediction"] == data["position_target"]) & (
                data["logsta_prediction"] == data["logsta_target"])
    print("Accuracy: {}     Correct: {}     Total: {}".format(100 * len(data[data["match"] == True]) / len(data),
                                                              len(data[data["match"] == True]), len(data)))


if __name__ == '__main__':
    position_prediction_file = "../../output/predictions/new-dataset/positions.csv"
    position_label_file = "../../datasets/stage1/new-test.csv"
    logsta_prediction_file = "../../output/predictions/new-dataset/statements_beam_search.csv"
    logsta_label_file = "../../datasets/stage2/new-test.csv"

    print("-" * 100)
    print(position_prediction_file)
    print(logsta_prediction_file)
    cal_position_metrics(position_prediction_file, position_label_file)
    cal_level_metrics(logsta_prediction_file, logsta_label_file)
    cal_message_metrics(logsta_prediction_file, logsta_label_file)
    cal_all_metrics(position_prediction_file, position_label_file, logsta_prediction_file, logsta_label_file)