# coding:utf-8
# json文件的读取和写入操作

import json

json_data = open("./data/data-text.json").read()
data = json.loads(json_data)

for item in data:
    print(item, ":", data[item])
