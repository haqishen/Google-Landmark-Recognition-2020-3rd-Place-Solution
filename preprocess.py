import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold


def main():
    df_train = pd.read_csv('train_0.csv')

    skf = StratifiedKFold(5, shuffle=True, random_state=233)

    df_train['fold'] = -1
    for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['landmark_id'])):
        df_train.loc[valid_idx, 'fold'] = i

    landmark_id2idx = {landmark_id: idx for idx, landmark_id in enumerate(sorted(df_train['landmark_id'].unique()))}
    with open('idx2landmark_id.pkl', 'wb') as fp:
        pickle.dump(landmark_id2idx, fp)


if __name__ == '__main__':
    main()
