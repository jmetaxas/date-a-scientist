import time
from data.load_dataframe import load_profiles
from utils.models_utils import lr, knr

start_time = time.time()
profiles = load_profiles(income=True)

"""
Age (Regression)
"""
# print(profiles.corrwith(profiles['age']))

current_guess = ['age']
all_features = [
    'high_income', 'middle_income', 'low_income',
    'has_high_academic_degree', 'has_graduated', 'is_studying',
    'drinks_code', 'drugs_code', 'smokes_code',
    'has_kids', 'has_no_kids', 'wants_kids', 'doesnt_want_kids',
]
lr(profiles, all_features, current_guess)
# knr(profiles, all_features, current_guess, 41, plot_best_k=1)

print('')
print("%s seconds" % (time.time() - start_time))
