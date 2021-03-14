import os
import shutil
import json
from datetime import datetime
from os.path import join as opj
import global_var


def backup_file(src, dst):
    if os.path.exists(opj(src, 'nobackup')):
        return
    if len([k for k in os.listdir(src) if k.endswith('.py') or k.endswith('.sh')]) == 0:
        return
    if not os.path.isdir(dst):
        os.makedirs(dst)
    all_files = os.listdir(src)
    for fname in all_files:
        fname_full = opj(src, fname)
        fname_dst = opj(dst, fname)
        if os.path.isdir(fname_full):
            backup_file(fname_full, fname_dst)
        elif fname.endswith('.py') or fname.endswith('.sh'):
            shutil.copy(fname_full, fname_dst)


def prepare_log_dir(log_name):
    if len(log_name) == 0:
        log_name = datetime.now().strftime("%b%d_%H%M%S")

    log_dir = os.path.join(global_var.LOG_DIR, log_name)
    if not os.path.exists(log_dir):
        print('making %s' % log_dir)
        os.makedirs(log_dir)
    else:
        print('log_dir({}) already exists'.format(log_dir))

    backup_dir = opj(log_dir, 'code')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    backup_file(global_var.ROOT_PATH, backup_dir)
    print("Backup code in {}".format(backup_dir))
    return log_dir


def save_params(log_dir, params, save_name="params"):
    same_num = 1
    while os.path.exists(save_name):
        save_name = save_name + "({})".format(same_num)
    with open(os.path.join(log_dir, save_name+".json"), 'w') as f:
        json.dump(params, f)