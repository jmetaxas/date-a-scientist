import numpy as np


def prepare_data(df, data_to_analyse):
    columns = []

    for group in data_to_analyse:
        if group == 'sex':
            df = build_sex_code(df)
            columns += ['sex_code']

        elif group == 'age_code':
            df = build_age_code(df)
            columns += ['age_code']

        elif group == 'orientation':
            df = build_binary_orientation(df)
            columns += ['is_straight', 'is_gay_bi']

        elif group == 'ethnicity':
            df = build_binary_ethnicity(df)
            columns += ['is_white', 'is_asian', 'is_latin', 'is_black',
                        'is_islander', 'is_native', 'is_middle_eastern', 'is_indian']

        elif group == 'religion':
            df = build_religion_binary(df)
            columns += ['is_agnostic', 'is_catholic', 'is_atheist', 'is_non_catholic_christian',
                        'is_jewish', 'is_buddhist', 'is_hindu', 'is_muslim']

        elif group == 'income':
            df = build_income_binary(df)
            columns += ['high_income', 'middle_income', 'low_income']

        elif group == 'income_code':
            df = build_income_code(df)
            columns += ['income_code']

        elif group == 'career':
            df = build_binary_career_field(df)
            columns += ['stem_career', 'health_career', 'law_career', 'artistic_career',
                        'education_career', 'business_career', 'financial_career']

        elif group == 'academic_degree':
            df = build_binary_academic_degree(df)
            columns += ['has_high_academic_degree', 'has_graduated', 'is_studying']

        elif group == 'body_type':
            df = build_binary_body_type(df)
            columns += ['has_chubby_body_type', 'has_fit_body_type', 'has_thin_body_type', 'has_average_body_type']

        elif group == 'diet':
            df = build_diet_binary(df)
            columns += ['eats_anything', 'eats_vegetarian', 'eats_vegan',
                        'eats_kosher', 'eats_halal']

        elif group == 'drinks_drugs_smokes':
            df = build_drinks_drugs_smokes_code(df)
            columns += ['drinks_code', 'drugs_code', 'smokes_code']

        elif group == 'offspring':
            df = build_offspring_binary(df)
            columns += ['has_kids', 'has_no_kids', 'wants_kids', 'doesnt_want_kids']

        elif group == 'pets':
            df = build_binary_pets(df)
            columns += ['has_cats', 'has_dogs', 'likes_cats',
                        'likes_dogs', 'dislikes_cats', 'dislikes_dogs']

        elif group == 'sign':
            df = build_binary_sign(df)
            columns += ['is_gemini', 'is_scorpio', 'is_leo', 'is_libra', 'is_taurus', 'is_cancer',
                        'is_pisces', 'is_sagittarius', 'is_virgo', 'is_aries', 'is_aquarius', 'is_capricorn']

        elif group == 'essay':
            df = stats_essay(df)
            columns += ['essay_len', 'essay_count_words', 'essay_words_mean_length']
            # df = stats_per_essay(df)
            # columns += ['essay0_length', 'essay0_count_words', 'essay0_words_mean_length',
            #             'essay1_length', 'essay1_count_words', 'essay1_words_mean_length',
            #             'essay2_length', 'essay2_count_words', 'essay2_words_mean_length',
            #             'essay3_length', 'essay3_count_words', 'essay3_words_mean_length',
            #             'essay4_length', 'essay4_count_words', 'essay4_words_mean_length',
            #             'essay5_length', 'essay5_count_words', 'essay5_words_mean_length',
            #             'essay6_length', 'essay6_count_words', 'essay6_words_mean_length',
            #             'essay7_length', 'essay7_count_words', 'essay7_words_mean_length',
            #             'essay8_length', 'essay8_count_words', 'essay8_words_mean_length',
            #             'essay9_length', 'essay9_count_words', 'essay9_words_mean_length']
        else:
            df = df

    return df, columns


def build_sex_code(df):
    df["sex_code"] = df.sex.map({
        "m": 0, "f": 1
    })

    # print(df.sex_code.value_counts())
    return df


def build_binary_orientation(df):
    df["is_straight"] = df.apply(lambda row: 1 if (
            row['orientation'] == 'straight'
    ) else 0, axis=1)

    df["is_gay_bi"] = df.apply(lambda row: 1 if (
            row['orientation'] == 'gay' or
            row['orientation'] == 'bisexual'
    ) else 0, axis=1)

    # print(df.is_straight.value_counts())
    return df


def build_binary_ethnicity(df):
    ethnicities = {
        'is_white': 'white',
        'is_asian': 'asian',
        'is_latin': 'hispanic / latin',
        'is_black': 'black',
        'is_islander': 'pacific islander',
        'is_native': 'native american',
        'is_middle_eastern': 'middle eastern',
        'is_indian': 'indian'
    }

    df['ethnicity'] = df['ethnicity'].replace(np.nan, '', regex=True)

    for key, ethnicity in ethnicities.items():
        df[key] = df.apply(lambda row: 1 if (
                ethnicity in row.ethnicity
        ) else 0, axis=1)

    # print(df.is_white.value_counts())

    return df


def build_religion_binary(df):
    religions = {
        'is_agnostic': 'agnosticism',
        'is_catholic': 'catholicism',
        'is_atheist': 'atheism',
        'is_non_catholic_christian': 'christianity',
        'is_jewish': 'judaism',
        'is_buddhist': 'buddhism',
        'is_hindu': 'hinduism',
        'is_muslim': 'islam'
    }

    df['religion'] = df['religion'].replace(np.nan, '', regex=True)

    for key, religion in religions.items():
        df[key] = df.apply(lambda row: 1 if (
                religion in row.religion
        ) else 0, axis=1)

    # print(df.is_catholic.value_counts())

    return df


def build_income_binary(df):
    df["high_income"] = df.apply(lambda row: 1 if (
            row['income'] >= 80000
    ) else 0, axis=1)

    df["middle_income"] = df.apply(lambda row: 1 if (
            30000 < row['income'] < 80000
    ) else 0, axis=1)

    df["low_income"] = df.apply(lambda row: 1 if (
            0 <= row['income'] <= 30000
    ) else 0, axis=1)

    return df


def build_binary_career_field(df):
    df['job'] = df['job'].replace(np.nan, '', regex=True)

    df['stem_career'] = df.apply(lambda row: 1 if (
            row.job == 'science / tech / engineering' or
            row.job == 'computer / hardware / software'
    ) else 0, axis=1)

    df['health_career'] = df.apply(lambda row: 1 if (
            row.job == 'medicine / health'
    ) else 0, axis=1)

    df['law_career'] = df.apply(lambda row: 1 if (
            row.job == 'law / legal services'
    ) else 0, axis=1)

    df['artistic_career'] = df.apply(lambda row: 1 if (
            row.job == 'artistic / musical / writer'
    ) else 0, axis=1)

    df['education_career'] = df.apply(lambda row: 1 if (
            row.job == 'education / academia'
    ) else 0, axis=1)

    df['business_career'] = df.apply(lambda row: 1 if (
            row.job == 'sales / marketing / biz dev'
    ) else 0, axis=1)

    df['financial_career'] = df.apply(lambda row: 1 if (
            row.job == 'banking / financial / real estate'
    ) else 0, axis=1)

    return df


def build_binary_academic_degree(df):
    df['education'] = df['education'].replace(np.nan, '', regex=True)

    df['has_high_academic_degree'] = df.apply(lambda row: 1 if (
            row.education == 'graduated from masters program' or
            row.education == 'graduated from ph.d program' or
            row.education == 'graduated from law school' or
            row.education == 'graduated from med school' or
            row.education == 'graduated from space camp'
    ) else 0, axis=1)

    df['has_graduated'] = df.apply(lambda row: 1 if (
            "graduated" in row.education
    ) else 0, axis=1)

    df['is_studying'] = df.apply(lambda row: 1 if (
            "working" in row.education
    ) else 0, axis=1)

    # print(df.has_high_academic_degree.value_counts())
    return df


def build_binary_body_type(df):
    df['has_chubby_body_type'] = df.apply(lambda row: 1 if (
                                row.body_type == 'overweight' or
                                row.body_type == 'full figured' or
                                row.body_type == 'curvy' or
                                row.body_type == 'a little extra'
                            ) else 0, axis=1)

    df['has_fit_body_type'] = df.apply(lambda row: 1 if (
                                row.body_type == 'fit' or
                                row.body_type == 'athletic' or
                                row.body_type == 'jacked'
                            ) else 0, axis=1)

    df['has_thin_body_type'] = df.apply(lambda row: 1 if (
                                row.body_type == 'skinny' or
                                row.body_type == 'thin' or
                                row.body_type == 'used up'
                            ) else 0, axis=1)

    df['has_average_body_type'] = df.apply(lambda row: 1 if (
                                row.body_type == 'average'
                            ) else 0, axis=1)

    return df


def build_diet_binary(df):
    df['eats_anything'] = df.apply(lambda row: 1 if (
            row.diet == 'mostly anything' or
            row.diet == 'anything' or
            row.diet == 'strictly anything'
    ) else 0, axis=1)

    df['eats_vegetarian'] = df.apply(lambda row: 1 if (
            row.diet == 'mostly vegetarian' or
            row.diet == 'vegetarian' or
            row.diet == 'strictly vegetarian'
    ) else 0, axis=1)

    df['eats_vegan'] = df.apply(lambda row: 1 if (
            row.diet == 'mostly vegan' or
            row.diet == 'vegan' or
            row.diet == 'strictly vegan'
    ) else 0, axis=1)

    df['eats_kosher'] = df.apply(lambda row: 1 if (
            row.diet == 'mostly kosher' or
            row.diet == 'kosher' or
            row.diet == 'strictly kosher'
    ) else 0, axis=1)

    df['eats_halal'] = df.apply(lambda row: 1 if (
            row.diet == 'mostly halal' or
            row.diet == 'halal' or
            row.diet == 'strictly halal'
    ) else 0, axis=1)

    return df


def build_drinks_drugs_smokes_code(df):
    df["drinks_code"] = df.drinks.map({
        "not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5
    })

    df["drugs_code"] = df.drugs.map({
        "never": 0, "sometimes": 1, "often": 2
    })

    df["smokes_code"] = df.smokes.map({
        "no": 0,
        "trying to quit": 1,
        "when drinking": 2,
        "sometimes": 3,
        "yes": 4
    })

    df.fillna({'drinks_code': 0,
               'drugs_code': 0,
               'smokes_code': 0
               }, inplace=True)

    # print(df.drinks_code.value_counts())
    return df


def build_offspring_binary(df):

    df['offspring'] = df['offspring'].replace(np.nan, '', regex=True)
    df['has_kids'] = df.apply(lambda row: 1 if (
            "has kids" in row.offspring or
            "has a kid" in row.offspring
    ) else 0, axis=1)
    #
    df['has_no_kids'] = df.apply(lambda row: 1 if (
            "doesn&rsquo;t have kids" in row.offspring
    ) else 0, axis=1)

    df['wants_kids'] = df.apply(lambda row: 1 if (
            "but wants them" in row.offspring or
            "but might want them" in row.offspring or
            "but might want more" in row.offspring or
            "wants kids" in row.offspring or
            "might want kids" in row.offspring or
            "and might want more" in row.offspring or
            "and wants more" in row.offspring
    ) else 0, axis=1)

    df['doesnt_want_kids'] = df.apply(lambda row: 1 if (
            "doesn&rsquo;t want kids" in row.offspring or
            "doesn&rsquo;t want any" in row.offspring or
            "but doesn&rsquo;t want more" in row.offspring
    ) else 0, axis=1)

    # print(df.has_kids.value_counts())

    return df


def build_binary_pets(df):
    pets = {
        'has_cats': "has cats",
        'has_dogs': "has dogs",
        'likes_cats': "likes cats",
        'likes_dogs': "likes dogs",
        'dislikes_cats': "dislikes cats",
        'dislikes_dogs': "dislikes dogs"
    }

    df['pets'] = df['pets'].replace(np.nan, '', regex=True)

    for key, pet in pets.items():
        df[key] = df.apply(lambda row: 1 if (
                pet in row.pets
        ) else 0, axis=1)

    return df


def build_binary_sign(df):
    signs = ['gemini', 'scorpio', 'leo', 'libra', 'taurus', 'cancer',
             'pisces', 'sagittarius', 'virgo', 'aries', 'aquarius', 'capricorn']

    df['sign'] = df['sign'].replace(np.nan, '', regex=True)

    for sign in signs:
        df['is_' + sign] = df.apply(lambda row: 1 if (
                sign in row.sign
        ) else 0, axis=1)

    return df


def stats_per_essay(df):
    essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]
    all_essays = df[essay_cols].replace(np.nan, '', regex=True)
    # all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
    #
    # df["essay_len"] = all_essays.apply(lambda x: len(x))

    for i in range(len(essay_cols)):
        df[essay_cols[i]+"_length"] = all_essays[essay_cols[i]].apply(lambda this_essay: len(this_essay))
        df[essay_cols[i]+"_count_words"] = all_essays[essay_cols[i]].apply(lambda this_essay: len(this_essay.split()))
        df[essay_cols[i]+"_words_mean_length"] = all_essays[essay_cols[i]].apply(
            lambda this_essay: np.mean(
                [len(word) for word in this_essay.split()]
            ))

        df.fillna({essay_cols[i]+'_words_mean_length': 0}, inplace=True)

    return df


def stats_essay(df):
    essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]
    all_essays = df[essay_cols].replace(np.nan, '', regex=True)
    all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

    df["essay_len"] = all_essays.apply(lambda x: len(x))
    df["essay_count_words"] = all_essays.apply(lambda this_essay: len(this_essay.split()))
    df["essay_words_mean_length"] = all_essays.apply(
        lambda this_essay: np.mean(
            [len(word) for word in this_essay.split()]
        ))

    df.fillna({'essay_words_mean_length': 0}, inplace=True)

    return df


def clean_income(df):
    df['income'] = df['income'].replace(-1, np.nan, regex=True)
    df = df.dropna(subset=["income"])

    return df


def build_income_code(df):
    conditions = [
        (df['income'] >= 20000) & (df['income'] <= 30000),
        (df['income'] > 30000) & (df['income'] < 80000),
        (df['income'] >= 80000)
    ]
    choices = [0, 1, 2]
    df["income_code"] = np.select(conditions, choices)

    return df


def build_age_code1(df):
    conditions = [
        (df['age'] >= 0) & (df['age'] <= 24),
        (df['age'] > 24) & (df['age'] <= 30),
        (df['age'] > 30) & (df['age'] <= 40),
        (df['age'] > 40)
    ]
    choices = [0, 1, 2, 3]
    df["age_code"] = np.select(conditions, choices)

    return df


def build_age_code(df):
    conditions = [
        (df['age'] >= 0) & (df['age'] <= 30),
        (df['age'] > 30)
    ]
    choices = [0, 1]
    df["age_code"] = np.select(conditions, choices)

    return df
