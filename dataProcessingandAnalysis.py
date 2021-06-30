# encoding: utf-8
# Author  : LouXM
# Datetime: 2021/6/28 10:38

# 导入相关模块
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False      # 中文
plt.rcParams['font.sans-serif'] = ['SimHei']    # 负号


data = pd.read_excel('./Movie.xlsx')
print(data.shape)
# print(data.columns)
# print(data.head())
print(data.isnull().sum())
# 缺失值处理，直接干掉，简单粗暴
rows_without_missing_data = data.dropna()
# 原来4937条数据，处理后3670条数据，能接受，不能接受也就这样了
print(rows_without_missing_data.shape)
# print(rows_without_missing_data.isnull().sum())
print(rows_without_missing_data['电影年份'].unique())
# 总共74年
print(len(rows_without_missing_data['电影年份'].unique()))
print(rows_without_missing_data['分级'].value_counts())
# 总共12个级别
print(len(rows_without_missing_data['分级'].unique()))

# 将电影类型切分开来
# 创建电影类型集合
genres = set()
for m in rows_without_missing_data['电影类型']:
    genres.update(g for g in m.split('|'))
genres = sorted(genres)
print(genres)
# 共22中电影类型
print(len(genres))
# 创建一个新的dataframe
genres_df = pd.DataFrame()
for genre in genres:
    genres_df[genre] = [genre in movie.split('|') for movie in rows_without_missing_data['电影类型']]
    rows_without_missing_data[genre] = [genre in movie.split('|') for movie in rows_without_missing_data['电影类型']]
print(genres_df.head(2))

# (后来发现字典没啥用基本上，就一直dataframe就ok，害，不删了)
# 统计电影类型和数量的字典
dictGenres = {}
# 统计一下每种类型共有多少部电影
for genre in genres:
    # print(rows_without_missing_data[genre].value_counts()[genre == True])
    dictGenres[genre] = rows_without_missing_data[genre].value_counts()[genre == True]
for kv in dictGenres.items():
    print(kv)

dictGenres_ = sorted(dictGenres.items(), key=lambda item:item[1], reverse=True)
print(dictGenres_)

genres_df['电影年份'] = rows_without_missing_data['电影年份']
genre_by_year = genres_df.groupby('电影年份').sum()
print(genre_by_year.head(2))

genre_sum = genre_by_year.sum().sort_values(ascending=False)
print(genre_sum)
# 统计每类电影对应发行的年份
kind = []
for genre in genres:
    kind.append(genre)
    # print(kind)
print(kind)
# print(rows_without_missing_data[rows_without_missing_data['Action']==True]['电影年份'])

# 没啥用
kind_year = []
count = 0
for key in dictGenres.keys():
    kind_year.append(dict.fromkeys(key, rows_without_missing_data[rows_without_missing_data[key]==True]['电影年份']))
    count += 1
print("手动分割线")
print(count)
# print(kind_year)

# 创建电影类型-利润的dataframe
profit_df = pd.DataFrame()
rows_without_missing_data['profit'] = rows_without_missing_data['票房'] - rows_without_missing_data['预算']
profit_df = pd.concat([genres_df.iloc[:,:-1],rows_without_missing_data['profit']],axis=1) # 将电影类型和利润数据加入心得dataframe
profit_by_genre = pd.Series(index=genres)
for genre in genres:
    profit_by_genre[genre] = profit_df.groupby(genre, as_index=False).profit.mean().loc[1, 'profit']
print(profit_by_genre)

# 创建国家的集合
country = set()
for x in rows_without_missing_data['国家/地区']:
    country.update(x.split("|"))
print(country)


# 创建国家dataframe
country_df = pd.DataFrame()
for c in country:
    country_df[c] = rows_without_missing_data['国家/地区'].str.contains(c).map(lambda x:1 if x else 0)
country_df = country_df.sum().sort_values(ascending=False)    # 计算各个国家电影数量
print(country_df)

# 创建电影类型与受欢迎程度的dataframe
popular_genre_df = pd.DataFrame()
popular_genre_df = pd.concat([genres_df.iloc[:,:-1], rows_without_missing_data['电影脸书粉丝数']], axis=1)
list = []
# 计算各类电影受欢迎程度的均值
for genre in genres:
    list.append(popular_genre_df.groupby(genre, as_index=False)['电影脸书粉丝数'].mean().loc[1, '电影脸书粉丝数'])
popular_by_genre = pd.DataFrame(index=genres)
popular_by_genre['popular_mean'] = list
print(popular_by_genre)

# 统计一下电影票房受哪些因素影响
print(rows_without_missing_data[['预算', '电影脸书粉丝数', '电影年份', '片长', 'IMDB评分', '评论人数', '票房']].corr())
# 0.094367  0.355798  0.047180  0.240714  0.217761  0.543473  1.000000

# 创建票房相关因素的dataframe
revenue_corr = rows_without_missing_data[['评论人数', '电影脸书粉丝数', '片长', '票房']]

# 统计一下情节关键字信息
keywords = set()
for keyword in rows_without_missing_data['情节关键字']:
    keywords.update(k for k in keyword.split('|'))
keywords = sorted(keywords)
print(keywords)


# 绘制情节关键字词云
wordcloud = WordCloud(width=1000,   # 图片的宽度
                      height=860,    # 高度
                      margin=2,    # 边距
                      background_color='black',   # 指定背景颜色
                      font_path='C:\Windows\Fonts\Sitka Banner\msyh.ttc'    # 指定字体文件，要有这个字体文件，自己随便想用什么字体，就下载一个，然后指定路径就ok了
                      )
wordcloud.generate(str(keywords))    # 分词
wordcloud.to_file('./figs/keywords.png')  # 保存到图片
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# 根据电影年份绘制直方图
sns.set(color_codes=True)
sns.set_style('white')
sns.set_palette('tab20')

fig, ax = plt.subplots()
rows_without_missing_data['电影年份'].hist(range=(1981, 2021), bins=40, color=(114/255, 158/255, 206/255))     # 从1981开始为了让图片更好看，81年之前太少了导致整体分布不好看
ax.set_title('Annual circulation of top box office movies in the world')

# 只保留底部坐标轴
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设定xy轴标签和网格，方便阅读数据
ax.set_yticklabels([0, 10, 20, 30, 40, 50])
ax.set_xticklabels(np.arange(1975, 2025, 5))
ax.grid(alpha=0.5)
plt.savefig("./figs/movie_released.png")
plt.show()

# 画一个年份和类型相关的图，折线或者饼状图(不同电影类型的年份累计分析)
# 根据电影类型统计数据作图

fig = plt.figure(figsize=(12, 6))
plt.plot(genre_by_year, label=genre_by_year.columns)
plt.legend(genre_by_year)
plt.xticks(range(1910, 2021, 10))
plt.title("Film type changing trend over time", fontsize=20)
plt.xlabel('year', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.savefig("./figs/genre_by_year.png")
plt.show()

# 电影类型数量分布直方图
genre_sum.plot.barh(label='genre', figsize=(12, 6))
plt.title("Distribution of film types", fontsize=20)
plt.xlabel("counts", fontsize=20)
plt.ylabel("genre", fontsize=20)
plt.savefig("./figs/genre_counts.png")
plt.show()

# 各类电影平均收益分布图
profit_by_genre.sort_values().plot.barh(label='genre', figsize=(12, 6))
plt.title("Profit distribution of film types", fontsize=20)
plt.xlabel('profit', fontsize=20)
plt.ylabel('genre', fontsize=20)
plt.savefig("./figs/genre_profit.png")
plt.show()

# 电影产地分布图
rate = country_df/country_df.sum()
others = 0.01
rate1 = rate[rate >= others]
rate1['others'] = rate[rate < others].sum()   # 把比例中小于1%的都放入others中
explode = (rate1 >= 0.04)/20 + 0.02   # 占比大于4%的向外延伸
rate1.plot.pie(autopct='%1.1f%%', figsize=(10, 10), explode=explode, label="")
plt.title("Distribution map of film producing areas", fontsize=20)
plt.savefig("./figs/film_producing_areas.png")
plt.show()

# 受众喜好
# 1.电影类型与受欢迎程度
popular_by_genre.sort_values(by='popular_mean').plot.barh(label='genre', figsize=(14, 8))
plt.title("Popularity distribution of film types", fontsize=20)
plt.legend(loc='best')
plt.xlabel('popularity', fontsize=20)
plt.ylabel('genre', fontsize=20)
plt.savefig("./figs/popularity_genre.png")
plt.show()

# 2.电影时长与受欢迎程度
area = np.pi * 4 ** 2
plt.scatter(rows_without_missing_data['片长'], rows_without_missing_data['电影脸书粉丝数'], s=area, c='#00CED1')
plt.title("Popularity distribution of movie duration", fontsize=20)
plt.xlabel('duration', fontsize=20)
plt.ylabel('popularity', fontsize=12)
plt.savefig("./figs/popularity_duration.png")
plt.show()

# 3.电影评分与受欢迎程度
plt.scatter(rows_without_missing_data['IMDB评分'], rows_without_missing_data['电影脸书粉丝数'], s=area, c='#DC143C')
plt.title("Movie ratings popularity distribution", fontsize=20)
plt.xlabel('rating', fontsize=15)
plt.ylabel('popularity', fontsize=12)
plt.savefig("./figs/popularity_rating.png")
plt.show()

# 票房与哪些因素最相关
fig = plt.figure(figsize=(18, 6))
# revenue_corr = rows_without_missing_data[['评论人数', '电影脸书粉丝数', '片长', '票房']]
# 1.电影票房与评论人数相关性散点图及线性回归线
ax1 = plt.subplot(1, 3, 1)
ax1 = sns.regplot(x='评论人数', y='票房', data=revenue_corr, x_jitter=.1, color='y')
ax1.text(0, 2.5e9, 'r=0.54', fontsize=18)
plt.title('Comments and box office', fontsize=15)
plt.xlabel("comments", fontsize=15)
plt.ylabel("box office")

# 2.电影票房与电影脸书粉丝数相关性散点图及线性回归线
ax2 = plt.subplot(1, 3, 2)
ax2 = sns.regplot(x='电影脸书粉丝数', y='票房', data=revenue_corr, x_jitter=.1, color='g')
ax2.text(0, 3.6e9, 'r=0.36', fontsize=18)
plt.title('Facebook fans and box office', fontsize=15)
plt.xlabel("fans", fontsize=15)
plt.ylabel("box office")

# 3.电影票房与片长相关性散点图及线性回归线
ax3 = plt.subplot(1, 3, 3)
ax3 = sns.regplot(x='片长', y='票房', data=revenue_corr, x_jitter=.1, color='r')
ax3.text(0, 2.5e9, 'r=0.24', fontsize=18)
plt.title('Film length and box office', fontsize=15)
plt.xlabel("duration", fontsize=15)
plt.ylabel("box office")
plt.savefig("./figs/factors_affecting_box_office.png")
plt.show()

# 最佳导演、演员..
# TODO ? Maybe..

