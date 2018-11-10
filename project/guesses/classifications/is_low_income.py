import time
from data.load_dataframe import load_profiles
from utils.models_utils import svc, knc, nbc

start_time = time.time()
profiles = load_profiles(income=True)

"""
Low Income (Classification)
"""
# print(profiles.corrwith(profiles['low_income']))
current_guess = ['low_income']
all_features = [
    'age', 'sex_code', 'height',
    'is_straight', 'is_gay_bi',
    'has_high_academic_degree', 'has_graduated', 'is_studying',
    'has_chubby_body_type', 'has_fit_body_type',
    'stem_career', 'education_career', 'financial_career',
    'drinks_code', 'drugs_code', 'smokes_code',
]
# svc(profiles, all_features, current_guess, vector_kernel='rbf', vector_c=4, vector_gamma=8, show_report=True)
knc(profiles, all_features, current_guess, 17, show_report=True, plot_best_k=1)
# nbc(profiles, all_features, current_guess)

print('')
print("%s seconds" % (time.time() - start_time))
