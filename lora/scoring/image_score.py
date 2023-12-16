import argparse
from sklearn.metrics import roc_auc_score

def main(args):

    y_true = [0, 0, 1, 1]
    y_predict = [0, 0, 0, 0]
    score = roc_auc_score(y_true, y_predict)
    print(score)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)