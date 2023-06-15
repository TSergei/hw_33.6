import json
import dill
import os
import pandas as pd
path = os.environ.get('PROJECT_PATH', '..')


def predict():
    # загружаем обученную модель
    model_pkl = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{model_pkl[0]}', 'rb') as f:
        model = dill.load(f)

    # создаём датафрейм для предсказания
    files = os.listdir(f'{path}/data/test')
    dict_for_pred = []
    for file_js in files:
        with open(f'{path}/data/test/{file_js}', 'rb') as f:
            data = json.load(f)
            dict_for_pred.append(data)
    df = pd.DataFrame(dict_for_pred)

    # делаем предикт для всего датафрейма
    df['pred_cat_price'] = model.predict(df)

    # сохраняем в csv-файл
    df[['id', 'pred_cat_price']].to_csv(f'{path}/data/predictions/df_preds.csv')


if __name__ == '__main__':
    predict()
