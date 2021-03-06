# BigDataAnalysis
## 数据集描述
这是一份从IMDB网站上提取的近100年间，65个国家共4937部电影的信息，共包括27个维度。该数据中的空值可以当作缺失值。某些维度的零值也可当作缺失值，会在各个维度的说明中标注。  
1.脸书粉丝(L-P) :该数据中包含导演/电影/1-3号演员脸书粉丝信息，对于以上所有维度，可以将零值当作缺失值。  
2.票房(T):该电影在美国的票房，货币类型为USD。  
3.预算(U):所有US地区的电影，预算的货币类型为USD。对于非US地区的电影，货币类型以IMDB页面信息为主。对于同一地区，电影的预算货币类型不一定相同。  
4.海报中人物数量(V):零值代表海报中无可识别的人物/演员。  
## 数据处理
为了简化工作量，将缺失值直接干掉，简单粗暴。原数据4937条，处理后3670条。
## 数据分析
### 电影情节关键字词云
![](figs/keywords.png)
### 根据电影年份绘制直方图
![](figs/movie_released.png)
### 根据电影类型统计数据作图
![](figs/genre_by_year.png)
### 电影类型数量分布直方图
![](figs/genre_counts.png)
### 各类电影平均收益分布图
![](figs/genre_profit.png)
### 电影产地分布图
![](figs/film_producing_areas.png)
### 受众喜好
#### 1.电影类型与受欢迎程度
![](figs/popularity_genre.png)
#### 2.电影时长与受欢迎程度
![](figs/popularity_duration.png)
#### 3.电影评分与受欢迎程度
![](figs/popularity_rating.png)

### 票房受影响因素
#### 1.电影票房与评论人数相关性
#### 2.电影票房与电影脸书粉丝数相关性
#### 3.电影票房与片长相关性
![](figs/factors_affecting_box_office.png)

## 参考文献
[blog1](https://www.jianshu.com/p/72eb16739fc5)  
[blog2](https://www.jianshu.com/p/a1fee4b3b5b1?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation)
