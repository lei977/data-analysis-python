## tensorflow算法的一般流程

### 1、导入/生成样本数据集

### 2、转换和归一化数据
TensorFlow具有内建函数来归一化数据：
data=tf.nn.batch_norm_with_global_normalization(...)

### 3、划分样本数据集为训练样本集、测试样本集和验证样本集

### 4、设置机器学习参数
learning_rate=0.01
batch_size=100
iterations=1000

### 5、初始化变量和占位符
在求解最优化过程中，TensorFlow通过占位符获取数据，并调整变量和权重/偏差。
TensorFlow指定数据大小和数据类型初始化变量和占位符。
a_var=tf.constant(42)
x_input=tf.placeholder(tf.float32,[None,input_size])
y_input=tf.placeholder(tf.float32,[None,num_classes])


### 6、定义模型结构
在获取样本数据集、初始化变量和占位符后，开始定义机器学习模型。
TensorFlow通过选择操作、变量和占位符的值来构建计算图。
例如简单线性模型：
y_pred=tf.add(tf.mul(x_input,weight_matrix),b_matrix)

### 7、声明损失函数
loss=tf.reduce_mean(tf.square(y_actual-y_pred))

### 8、初始化模型和训练模型
TensorFlow创建计算图实例，通过占位符赋值，维护变量的状态信息。
with tf.Session(graph=graph) as session:
    ...
    session.run(...)
    ...
    
或者：
session=tf.Session(graph=graph)
session.run(...)

### 9、评估机器学习模型

### 10、调优超参数

### 11、发布/预测结果