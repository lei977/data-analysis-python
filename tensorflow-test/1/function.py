# coding:utf-8

# 函数、模块的基本使用
import time

# 显示当前时间
print(time.asctime())


# 定义函数
def add(a, b):
    return a + b


# 函数调用
c = add(1, 5)
print(c)

# 内建函数：Python解释器自带的函数
print(abs(-10))


# 类定义及使用
class Animals:
    def breath(self):
        print("breathing")

    def move(self):
        print("moving")

    def eat(self):
        print("eating food")


class Mammals(Animals):
    pass


class Cats(Mammals):
    pass
