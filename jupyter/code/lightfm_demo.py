#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lightfm import LightFM
from lightfm.data import Dataset

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)

SELECTED_FEATURES = ['location_region_code', 'industry_code']
NUM_OF_ROWS = None

def get_industry_code_1(industry_code_2):
    if (industry_code_2 == None): return None
    ic_int = int(industry_code_2[4:])

    if (ic_int == 0): return 'IND_X'
    elif (ic_int >= 1 and ic_int <= 3): return 'IND_A'
    elif (ic_int >= 5 and ic_int <= 9): return 'IND_B'
    elif (ic_int >= 10 and ic_int <= 33): return 'IND_C'
    elif (ic_int == 35): return 'IND_D'
    elif (ic_int >= 36 and ic_int <= 39): return 'IND_E'
    elif (ic_int >= 41 and ic_int <= 43): return 'IND_F'
    elif (ic_int >= 45 and ic_int <= 47): return 'IND_G'
    elif (ic_int >= 49 and ic_int <= 53): return 'IND_H'
    elif (ic_int >= 55 and ic_int <= 56): return 'IND_I'
    elif (ic_int >= 58 and ic_int <= 63): return 'IND_J'
    elif (ic_int >= 64 and ic_int <= 66): return 'IND_K'
    elif (ic_int == 68): return 'IND_L'
    elif (ic_int >= 69 and ic_int <= 75): return 'IND_M'
    elif (ic_int >= 77 and ic_int <= 82): return 'IND_N'
    elif (ic_int == 84): return 'IND_O'
    elif (ic_int == 85): return 'IND_P'
    elif (ic_int >= 86 and ic_int <= 88): return 'IND_Q'
    elif (ic_int >= 90 and ic_int <= 93): return 'IND_R'
    elif (ic_int >= 94 and ic_int <= 96): return 'IND_S'
    elif (ic_int >= 97 and ic_int <= 98): return 'IND_T'
    elif (ic_int == 99): return 'IND_U'
    else:
        print('OBS! Found invalid industry code: %s', industry_code_2)
        return None


def load_data():
    company_df = pd \
        .read_csv('data/prod_data_companies_2021_08_22.csv',
                  delimiter='\t',
                  dtype={
                      'business_id': 'string',
                      'company_name': 'string',
                      'company_form': 'string',
                      'company_form_code': 'string',
                      'location_region': 'string',
                      'location_region_code': 'string',
                      'location_municipality': 'string',
                      'location_municipality_code': 'string',
                      'industry_code': 'string',
                      'company_status': 'string',
                      'company_status_code': 'string',
                      'personnel_class': 'string'
                  },
                  nrows=NUM_OF_ROWS
                  )

    # company_df['industry_code_1'] = company_df.apply(lambda row: get_industry_code_1(row['industry_code_2']), axis=1)

    item_ids = list(company_df['business_id'].values)

    item_feature_labels = set()
    for feature in SELECTED_FEATURES:
        item_feature_labels |= set(company_df[feature].unique())

    item_feature_labels = list(item_feature_labels)

    item_features = [(company['business_id'], [company[feature] for feature in SELECTED_FEATURES])
                     for company in company_df.to_dict(orient='records')]

    interaction_data = load_user_interaction_data(item_ids)

    return {
        'company_df': company_df,
        'item_ids': item_ids,
        'item_feature_labels': item_feature_labels,
        'item_features': item_features,
        'user_ids': interaction_data['user_ids'],
        'user_feature_labels': [],
        'user_feature_data': [],
        'interactions': interaction_data['interactions']
    }


def load_user_interaction_data(item_ids):
    user611_df = pd \
        .read_csv('../data/testiryhma-611-liikevaihto_yli_3M.csv',
                  delimiter=';',
                  converters={'Y-tunnus': lambda x: str(x)}) \
        .drop_duplicates('Y-tunnus', keep='first')

    user862_df = pd \
        .read_csv('../data/testiryhma-862-liikevaihto_yli_3M.csv',
                  delimiter=';',
                  converters={'Y-tunnus': lambda x: str(x)}) \
        .drop_duplicates('Y-tunnus', keep='first')

    interactions = [('611', bid) for bid in user611_df['Y-tunnus']] + [('862', bid) for bid in user862_df['Y-tunnus']]

    return {
        'user_ids': ['611', '862'],
        'interactions': list(filter(lambda interaction: interaction[1] in item_ids, interactions)) # filtteröi y-tunnukset, joita ei löydy testidatasta

    }


def parse_data_in_lightfm_format(data):
    dataset = Dataset(user_identity_features=False, item_identity_features=False)
    dataset.fit(users=data['user_ids'], items=data['item_ids'], item_features=data['item_feature_labels'])
    item_features = dataset.build_item_features(data['item_features'], normalize=True)
    (interactions, weights) = dataset.build_interactions(data['interactions'])

    return {
        'user_map': dataset.mapping()[0],
        'item_map': dataset.mapping()[2],
        'item_features': item_features,
        'interactions': interactions
    }


def print_sample_recommendations(model, lfm_data, data, user_ids):
    company_df = data['company_df']

    for user_id in user_ids:
        known_positives = [interaction[1] for interaction in filter(lambda x: x[0] == user_id, data['interactions'])]
        print("User %s:" % user_id)
        print("\tKnown positives:")

        for bid in known_positives[:3]:
            print("\t\t%s" % bid, company_df.loc[company_df['business_id'] == bid]['company_name'].item())

        lfm_user_id = lfm_data['user_map'][user_id]

        scores = model.predict([lfm_user_id], list(lfm_data['item_map'].values()),
                               item_features=lfm_data['item_features'])
        scores_df = pd.DataFrame.from_records(zip(list(lfm_data['item_map'].keys()), scores), columns=['business_id', 'scores'])
        merged_scores_df = pd.merge(scores_df, data['company_df'], on='business_id')

        print("\tTop hits:")
        print(merged_scores_df.sort_values('scores', 0, False).head(10))
        print("\n\n")


def initialize_model():
    data = load_data()
    lfm_data = parse_data_in_lightfm_format(data)

    model = LightFM(loss='warp')
    model.fit(lfm_data['interactions'], item_features=lfm_data['item_features'], epochs=5, verbose=True)

    print_sample_recommendations(model, lfm_data, data, ['611', '862'])
