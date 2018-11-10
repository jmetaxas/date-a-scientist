import time
from data.load_dataframe import load_profiles
from utils.models_utils import svc, knc, nbc

start_time = time.time()
profiles = load_profiles(income=True)

"""
Age classes (Classification)
"""
# print(profiles.age_code.value_counts())
# print(profiles.corrwith(profiles['age_code']))
# print(profiles.isna().any())

current_guess = ['age_code']
all_features = [
    'income',
    'high_income', 'middle_income', 'low_income',
    'has_high_academic_degree', 'has_graduated', 'is_studying',

    'is_agnostic', 'is_catholic', 'is_atheist', 'is_non_catholic_christian',
    'is_jewish', 'is_buddhist', 'is_hindu', 'is_muslim',

    'has_fit_body_type', 'has_chubby_body_type', 'has_thin_body_type', 'has_average_body_type',
    'eats_anything', 'eats_vegetarian', 'eats_vegan',
    'drinks_code', 'drugs_code', 'smokes_code',

    'has_kids', 'has_no_kids', 'wants_kids', 'doesnt_want_kids',
    'has_cats', 'has_dogs',

    'stem_career', 'health_career', 'law_career', 'artistic_career',
    'education_career', 'business_career', 'financial_career',

    'essay_len', 'essay_count_words', 'essay_words_mean_length',
]

mclasses=['0 (under 30)', '1 (over 30)']
svc(profiles, all_features, current_guess, vector_kernel='linear', vector_c=4, vector_gamma=8, show_report=True, show_matrix=True, matrix_classes=mclasses)
# knc(profiles, all_features, current_guess, 44, show_report=True, plot_best_k=1, show_matrix=False)
# nbc(profiles, all_features, current_guess)


print('')
print("%s seconds" % (time.time() - start_time))
