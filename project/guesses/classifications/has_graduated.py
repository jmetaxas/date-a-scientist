import time
from data.load_dataframe import load_profiles
from utils.models_utils import svc, knc, nbc

start_time = time.time()
profiles = load_profiles(income=True)

"""
Has gratuated (Classification)
"""
# print(profiles.corrwith(profiles['has_graduated']))
current_guess = ['has_graduated']
all_features = [
    'age',
    'income',
]
# svc(profiles, all_features, current_guess, vector_c=4, vector_gamma=8, show_report=True)
knc(profiles, all_features, current_guess, 22, show_report=True, plot_best_k=1)
# nbc(profiles, all_features, current_guess)

print('')
print("%s seconds" % (time.time() - start_time))
