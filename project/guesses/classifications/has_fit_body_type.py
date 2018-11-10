import time
from data.load_dataframe import load_profiles
from utils.models_utils import svc, knc, nbc

start_time = time.time()
profiles = load_profiles(income=True)

"""
Has fit body (Classification)
"""
# print(profiles.corrwith(profiles['has_fit_body_type']))
current_guess = ['has_fit_body_type']
all_features = [
    'sex_code', 'age', 'height',
    'is_straight', 'is_gay_bi',
    'eats_anything', 'eats_vegetarian', 'eats_vegan', 'eats_kosher', 'eats_halal',
    'drinks_code', 'drugs_code', 'smokes_code',
    'high_income', 'low_income',
    'has_high_academic_degree', 'has_graduated', 'is_studying',
]
# svc(profiles, all_features, current_guess, vector_kernel='rbf', vector_c=4, vector_gamma=8, show_report=True)
knc(profiles, all_features, current_guess, 58, show_report=True, plot_best_k=1)
# nbc(profiles, all_features, current_guess)

print('')
print("%s seconds" % (time.time() - start_time))
