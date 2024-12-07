import os

current_dir = os.path.dirname(os.path.abspath(__file__))

stella_path = os.path.join(current_dir, "../models/stella-base-zh-v2")
gte_path = os.path.join(current_dir, "../models/gte-base-zh")
dmeta_path = os.path.join(current_dir, "../models/Dmeta-embedding-zh")

# train_path = 'data/cmrc2018_train.json'
train_path = os.path.join(current_dir, "../data/cmrc2018_train.json")
trial_path = os.path.join(current_dir, "../data/cmrc2018_trial.json")
# val_path = 'data/cmrc2018_dev.json'

ques_path = os.path.join(current_dir, "../data/user_ques.xlsx")
test_path = os.path.join(current_dir, "../data/test.xlsx")
rating_path = os.path.join(current_dir, "../data/ratings.xlsx")
count_path = os.path.join(current_dir, "../data/word_count.xlsx")
