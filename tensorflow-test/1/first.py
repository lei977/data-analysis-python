# coding:utf-8

# helloworld程序
print('hello,world!')

# python运算符
print(4 * 8 + 2 * 3, 7 % 4, 5 / 2)

# 占位符%s
print("占位符：")
print('point = %s \n %s' % (0, 1))

# 列表
c = [1, 2, 3]
e = [1, 2, '4']

print("列表：")
print(c[1])
print(e)
print(c[0:2])
print(e[:])

# 元组
f = (1, 2, 3)

print("元组：")
print(f)

# 字典
dic = {1: "123", 'name': 'zhangsan'}

print("字典：")
print(dic["name"])

# 字典操作：修改、删除、插入
dic[1] = "999"
print("修改键‘1’：")
print(dic)

del dic['name']
print("删除键‘name’：")
print(dic)

dic['insert'] = '插入值'
print("插入新键值：")
print(dic)

# 条件语句1
if 1 > 0:
    print("条件语句输出1")

# 条件语句2
a = 10
if a > 10:
    print("条件语句输出2：", a, ">10")
else:
    print("条件语句输出2：", a, "!>10")

# 条件语句3
b = 3
if b == 1:
    print("对应条件1")
elif b == 2:
    print("对应条件2")
elif b == 3:
    print("对应条件3")
else:
    print("无对应值")

# 循环语句1
for i in range(1, 5):
    print("循环1输出值：", i)

# 循环语句2
mm = [1, 12, 123]
for m in mm:
    print("循环2输出值：", m)

# 循环语句3
n = 10
while n < 15:
    print("循环3输出值：", n)
    n += 1
    if n == 13:
        break
