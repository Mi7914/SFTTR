import matplotlib.pyplot as plt
import numpy as np

font = {'family':'Times New Roman','size':14}
# 生成三组随机数据，代表全班的语文、数学和英语分数
nscores = 60  # 生成 60 个学生的分数数据
chinese_scores = np.random.randint(50, 100, size=nscores)
math_scores = np.random.randint(40, 90, size=nscores)
english_scores = np.random.randint(60, 100, size=nscores)

# 创建画布和子图
fig, ax = plt.subplots()

# 绘制散点图，并指定不同颜色、大小和透明度
x = np.arange(nscores)
y = x
c1 = chinese_scores  # 颜色深浅与分数相关
c2 = math_scores
c3 = english_scores
ax.scatter(chinese_scores, math_scores, s=5)
ax.scatter(chinese_scores, english_scores,s=5)
ax.scatter(math_scores, english_scores, s=5)
plt.show()


# 设置坐标轴范围、刻度和标签字体样式
# ax.set_xlim(40, 100)
# ax.set_ylim(40, 110)
# ax.set_xticks(range(40, 110, 10))
# ax.set_yticks(range(40, 110, 10))
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.size'] = 20
# ax.set_xlabel('Chinese Scores', )
# ax.set_ylabel('Math Scores / English Scores')

# ax.legend(ncol=3,loc=1,prop=font)
# plt.tight_layout()
# plt.savefig('散点图.jpg')
# 展示图形
