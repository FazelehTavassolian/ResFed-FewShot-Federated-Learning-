from pathlib import Path
import shutil
import json
import numpy as np

n_episodes = 1000

data_root = Path('/home/jericho/Project/federated-meta-learning/Data/omniglot')
background = data_root.joinpath('images_background')
valid = data_root.joinpath('images_evaluation')

save_path = Path('/home/jericho/Project/federated-meta-learning/Data/omniglot-fs')
save_path.mkdir(parents=True, exist_ok=True)

# save background
tr_save = save_path.joinpath('train')
tr_save.mkdir(parents=True, exist_ok=True)
# for src_f in background.glob("*/*/*"):
#     dest_fol = f'{src_f.parent.parent.stem}-{src_f.parent.stem}'
#     dest_save = tr_save.joinpath(dest_fol)
#     dest_save.mkdir(parents=True, exist_ok=True)
#
#     shutil.copy(src_f, dest_save.joinpath(src_f.name))

# save val
vl_save = save_path.joinpath('val')
vl_save.mkdir(parents=True, exist_ok=True)
# for src_f in valid.glob("*/*/*"):
#     dest_fol = f'{src_f.parent.parent.stem}-{src_f.parent.stem}'
#     dest_save = vl_save.joinpath(dest_fol)
#     dest_save.mkdir(parents=True, exist_ok=True)
#     shutil.copy(src_f, dest_save.joinpath(src_f.name))


way5_shot1 = []
way5_shot5 = []
for ep_idx in range(n_episodes):
    all_parents = list(vl_save.glob("*"))
    # 5 Way 1 shot
    way_idx = np.random.choice(all_parents, 5, replace=False)
    select = {'Support': [], "Query": []}
    for src_f in way_idx:
        files = list(src_f.glob('*'))
        f_names = []
        for file in files[:1]:
            f_names.append(f'{file.parent.stem}/{file.name}')
        select['Support'].append(f_names)
        f_names = []
        end = min(len(files) - 1, 15)
        for file in files[1:end]:
            f_names.append(f'{file.parent.stem}/{file.name}')
        select['Query'].append(f_names)
    way5_shot1.append(select)

    # 5 Way 5 Shot
    way_idx = np.random.choice(all_parents, 5, replace=False)
    select = {'Support': [], "Query": []}
    for src_f in way_idx:
        files = list(src_f.glob('*'))
        f_names = []
        for file in files[:5]:
            f_names.append(f'{file.parent.stem}/{file.name}')
        select['Support'].append(f_names)
        f_names = []
        for file in files[5:]:
            f_names.append(f'{file.parent.stem}/{file.name}')
        select['Query'].append(f_names)
    way5_shot5.append(select)

with open(str(save_path.joinpath('val1000Episode_5_way_1_shot.json')), 'w') as f:
    json.dump(way5_shot1, f)

with open(str(save_path.joinpath('val1000Episode_5_way_5_shot.json')), 'w') as f:
    json.dump(way5_shot5, f)
