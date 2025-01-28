import dill
import json
import pandas as pd
import glob
import os

from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    mod = sorted(os.listdir(f'{path}/data/models'))

    with open(f'{path}/data/models/{mod[-1]}', 'rb') as file:
        model = dill.load(file)

    predict_df = pd.DataFrame(columns=['car_id', 'predict'])

    for datapath in glob.glob(f'{path}/data/test/*.json'):
        with open(datapath) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            X = {'car_id': df.id, 'predict': y}
            df1 = pd.DataFrame(X)
            predict_df = pd.concat([predict_df, df1], axis=0)
    print(predict_df)

    predict_df.to_csv(f'{path}/data/predictions/predict_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False)


if __name__ == '__main__':
    predict()
