# coding:utf-8
# csv文件的读取和写入操作

import csv

csv_data = open("./data/data-text.csv", "rb")
reader = csv.reader(csv_data)

for row in reader:
    print(row)
