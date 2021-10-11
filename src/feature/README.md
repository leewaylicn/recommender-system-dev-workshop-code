# Feature

online service to get item value feature

feature service 接收post请求，取得帖子的文本，然后缓存成文件。</br>
feature plugin 从缓存的文件中读，先进行数据预处理，然后一个一个batch依次获取bert base feature然后针对帖子的价值做排序。将帖子和得分存储起来。