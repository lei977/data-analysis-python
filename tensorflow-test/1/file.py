# coding:utf-8

# 文件读写操作

import pickle

# 变量存储操作

game_data = \
    {
        "position": "N2,e3",
        "pocket": ['keys', 'knife'],
        "money": 160
    }

save_file = open("./file/save.dat", "wb")
pickle.dump(game_data, save_file)
save_file.close()

# 变量读取操作

load_file = open("./file/save.dat", "rb")
load_game_data = pickle.load(load_file)
load_file.close()

print(load_game_data)

