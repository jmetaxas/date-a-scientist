import time
from data.load_dataframe import load_profiles
from utils.models_utils import lr, knr

start_time = time.time()
profiles = load_profiles(income=True)

"""
Income (Regression)
"""
# print(profiles.corrwith(profiles['income']))

current_guess = ['income']
all_features = [
    'age', 'height', 'sex_code',

    'is_white', 'is_asian', 'is_latin', 'is_black', 'is_islander',
    'is_native', 'is_middle_eastern', 'is_indian',

    'is_agnostic', 'is_catholic', 'is_atheist', 'is_non_catholic_christian',
    'is_jewish', 'is_buddhist', 'is_hindu', 'is_muslim',

    'stem_career', 'health_career', 'law_career', 'artistic_career',
    'education_career', 'business_career', 'financial_career',

    'has_fit_body_type', 'has_chubby_body_type', 'has_thin_body_type', 'has_average_body_type',

    'eats_anything', 'eats_vegetarian', 'eats_vegan',
    'eats_kosher', 'eats_halal',

    'drinks_code', 'drugs_code', 'smokes_code',

    'has_kids', 'has_no_kids', 'wants_kids', 'doesnt_want_kids',

    'has_high_academic_degree', 'has_graduated', 'is_studying',

    'essay_len', 'essay_count_words', 'essay_words_mean_length',
]
# lr(profiles, all_features, current_guess)
knr(profiles, all_features, current_guess, 26, plot_best_k=1)

print('')
print("%s seconds" % (time.time() - start_time))
