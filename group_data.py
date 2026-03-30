import json
import pickle
import csv
import os
from datetime import datetime
import math

from tqdm import tqdm
import dgl
import torch

#from feat_extraction.extract_video_feat import video2feat
from feat_extraction.extract_chinese_feat import txt2feat, txt2feat_bacth
from feat_extraction.extract_time_feat import time2feat
from data import find_file


DIVIDE_DATE = "2024-12-20"
graph_path = 'data/graph_with_labels.pkl'
# 创建异构图
# 读取 JSON 数据
with open("data/splited_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 计算天数差函数
def calculate_days_diff(time1, time2):
    time1 = datetime.strptime(time1.split()[0], "%Y-%m-%d")  
    time2 = datetime.strptime(time2.split()[0], "%Y-%m-%d")

    return (time2 - time1).days

def is_earlier(date1, date2):
    '''判断date1是否早于date2
    :param date1:
    :param date2:
    :return:
    '''
    time1 = datetime.strptime(date1.split()[0], "%Y-%m-%d")  
    time2 = datetime.strptime(date2.split()[0], "%Y-%m-%d")
    return time1 <= time2


# 存储节点信息的列表
video_info = []
des_info = []
title_info = []
topic_info = []
video_time_info = []
platform_info = []
comment_info = []
time_info = []
ctime_info = []
fans_info = []
likes_info = []
plays_info = []
shares_info = []
collections_info = []
comments_info = []

# 存储边的数据
edges_description_to_video = []
edges_title_to_video = []
edges_topic_to_video = []
edges_video_time_to_video = []
edges_platform_to_video = []
edges_comment_to_video = []
edges_time_to_video = []
edges_ctime_to_video = []
edges_fans_to_video = []
edges_likes_to_video = []
edges_plays_to_video = []
edges_shares_to_video = []
edges_collections_to_video = []
edges_comments_to_video = []
edges_video_to_video_history = []
edges_video_to_video_topic = []
edges_video_to_video_author = []

# 划分训练测试集的mask
train_mask = torch.zeros(len(data), dtype=torch.bool)
test_mask = torch.zeros(len(data), dtype=torch.bool)

# 存储topic节点的映射
topic_mapping = {}

# 存储platform节点的映射
platform_mapping = {}

#存储作者视频映射
author_videos = {}

#存储共同topic视频映射
topic_video_mapping = {}


# 存储视频数据
for video_id, video_data in tqdm(data.items()):
    # 视频节点
    video_node = {
        "id": int(video_id),  # 使用视频ID作为节点ID
        "url": video_data["url"],
        "影响力等级": video_data["影响力等级"],
        "平台": video_data["平台"]
    }
    video_info.append(video_node)

    # 划分训练测试集
    if not video_data['最终状态']:
        train_mask[int(video_id)] = is_earlier(video_data["发布时间"], DIVIDE_DATE)
        test_mask[int(video_id)] = not is_earlier(video_data["发布时间"], DIVIDE_DATE)

    # 描述节点
    des_node = {"id": len(des_info), "text": video_data["描述"]}
    des_info.append(des_node)
    edges_description_to_video.append((des_node["id"], int(video_id)))  

    # 标题节点
    title_node = {"id": len(title_info), "text": video_data["标题"]}
    title_info.append(title_node)
    edges_title_to_video.append((title_node["id"], int(video_id))) 
    
    # 视频时常节点
    video_time_node = {"id": len(video_time_info), "text": video_data["视频时长（秒）"]}
    video_time_info.append(video_time_node)
    edges_video_time_to_video.append((video_time_node["id"], int(video_id))) 

    # 平台节点
    platform_text = video_data["平台"]
    if platform_text not in platform_mapping:
        platform_node = {"id": len(platform_info), "text": platform_text}
        platform_info.append(platform_node)
        platform_mapping[platform_text] = platform_node["id"]
    edges_platform_to_video.append((platform_mapping[platform_text], int(video_id))) 

    # 评论节点
    for cid in video_data["评论"]:
        comment = video_data["评论"][cid]
        comment_node = {"id": len(comment_info), "text": comment["评论内容"], "time": comment["评论时间"]}
        comment_info.append(comment_node)
        edges_comment_to_video.append((comment_node["id"], int(video_id)))

    # 时间节点
    time_node = {"id": len(time_info), "text": video_data["发布时间"]}
    time_info.append(time_node)
    edges_time_to_video.append((time_node["id"], int(video_id))) 

    # 当前时间节点
    ctime_node = {"id": len(ctime_info), "text": video_data["当前时间"]}
    ctime_info.append(ctime_node)
    edges_ctime_to_video.append((ctime_node["id"], int(video_id))) 

    #粉丝数节点
    fans_node = {"id": len(fans_info), "text": video_data["当前粉丝量"]}
    fans_info.append(fans_node)
    edges_fans_to_video.append((fans_node["id"], int(video_id))) 

    # 点赞量节点
    likes_node = {"id": len(likes_info), "text": video_data["当前点赞量"]}
    likes_info.append(likes_node)
    edges_likes_to_video.append((likes_node["id"], int(video_id))) 

    # 播放量节点
    plays_node = {"id": len(plays_info), "text": video_data["当前播放量"]}
    plays_info.append(plays_node)
    edges_plays_to_video.append((plays_node["id"], int(video_id))) 

    # 分享量节点
    shares_node = {"id": len(shares_info), "text": video_data["当前分享量"]}
    shares_info.append(shares_node)
    edges_shares_to_video.append((shares_node["id"], int(video_id)))

    # 收藏量节点
    collections_node = {"id": len(collections_info), "text": video_data["当前收藏量"]}
    collections_info.append(collections_node)
    edges_collections_to_video.append((collections_node["id"], int(video_id)))

    # 评论量节点
    comments_node = {"id": len(comments_info), "text": video_data["当前评论量"]}
    comments_info.append(comments_node)
    edges_comments_to_video.append((comments_node["id"], int(video_id)))

    # 话题节点 - 去重处理
    topic_text = video_data["话题"]
    if topic_text not in topic_mapping:
        topic_node = {"id": len(topic_info), "text": topic_text}
        topic_info.append(topic_node)
        topic_mapping[topic_text] = topic_node["id"]
    edges_topic_to_video.append((topic_mapping[topic_text], int(video_id))) 

    # 历史视频判断
    for video_id2 in video_data['历史状态']:
        edges_video_to_video_history.append((int(video_id2), int(video_id)))

    # 更新作者发布视频列表
    if video_data['发布用户ID'] not in author_videos:
        author_videos[video_data['发布用户ID']] = [int(video_id)]
    else:
        author_videos[video_data['发布用户ID']].append(int(video_id))

    #判断相同视频话题
    if video_data['话题'] not in topic_video_mapping:
        topic_video_mapping[video_data['话题']] = [int(video_id)]  
    else:
        topic_video_mapping[video_data['话题']].append(int(video_id))      

if not os.path.exists(graph_path):
    for videos in author_videos.values():
        for i, video_id1 in enumerate(videos):
            for video_id2 in videos[i:]:
                if is_earlier(data[str(video_id1)]["当前时间"], data[str(video_id2)]["当前时间"]):
                    edges_video_to_video_author.append((video_id1, video_id2))
                if is_earlier(data[str(video_id2)]["当前时间"], data[str(video_id1)]["当前时间"]):
                    edges_video_to_video_author.append((video_id2, video_id1))

    for videos in topic_video_mapping.values():
        for i, video_id1 in enumerate(videos):
            for video_id2 in videos[i:]:
                if is_earlier(data[str(video_id1)]["当前时间"], data[str(video_id2)]["当前时间"]):
                    edges_video_to_video_topic.append((video_id1, video_id2))
                if is_earlier(data[str(video_id2)]["当前时间"], data[str(video_id1)]["当前时间"]):
                    edges_video_to_video_topic.append((video_id2, video_id1))        

    print("Constructing tensors...")
    # 构建异构图
    edges_description_to_video = torch.LongTensor(edges_description_to_video)
    edges_title_to_video = torch.LongTensor(edges_title_to_video)
    edges_topic_to_video = torch.LongTensor(edges_topic_to_video)
    edges_video_time_to_video = torch.LongTensor(edges_video_time_to_video)
    edges_platform_to_video = torch.LongTensor(edges_platform_to_video)
    edges_comment_to_video = torch.LongTensor(edges_comment_to_video)
    edges_time_to_video = torch.LongTensor(edges_time_to_video)
    edges_ctime_to_video = torch.LongTensor(edges_ctime_to_video)
    edges_fans_to_video = torch.LongTensor(edges_fans_to_video)
    edges_likes_to_video = torch.LongTensor(edges_likes_to_video)
    edges_plays_to_video = torch.LongTensor(edges_plays_to_video)
    edges_shares_to_video = torch.LongTensor(edges_shares_to_video)
    edges_collections_to_video = torch.LongTensor(edges_collections_to_video)
    edges_comments_to_video = torch.LongTensor(edges_comments_to_video)
    edges_video_to_video_history = torch.LongTensor(edges_video_to_video_history)
    edges_video_to_video_topic = torch.LongTensor(edges_video_to_video_topic)
    edges_video_to_video_author = torch.LongTensor(edges_video_to_video_author)

    print("Constructing graph...")
    g = dgl.heterograph({
        # 视频和其它节点的边关系 
        ("description", "is_description_of", "video"): (edges_description_to_video[:, 0], edges_description_to_video[:, 1]),
        
        ("title", "is_title_of", "video"): (edges_title_to_video[:, 0], edges_title_to_video[:, 1]),
        
        ("topic", "is_topic_of", "video"): (edges_topic_to_video[:, 0], edges_topic_to_video[:, 1]),

        #("content", "is_content_of", "video"): edges_content_to_video,
        
        ("video_time", "is_duration_time_of", "video"): (edges_video_time_to_video[:, 0], edges_video_time_to_video[:, 1]),

        ("platform", "is_platform_of", "video"): (edges_platform_to_video[:, 0], edges_platform_to_video[:, 1]),

        ("comment", "is_comment_of", "video"): (edges_comment_to_video[:, 0], edges_comment_to_video[:, 1]),

        ("time", "is_post_time_of", "video"): (edges_time_to_video[:, 0], edges_time_to_video[:, 1]),

        ("ctime", "is_current_time_of", "video"): (edges_ctime_to_video[:, 0], edges_ctime_to_video[:, 1]),

        ("fans", "is_fans_of", "video"): (edges_fans_to_video[:, 0], edges_fans_to_video[:, 1]),

        ("likes", "is_likes_of", "video"): (edges_likes_to_video[:, 0], edges_likes_to_video[:, 1]),

        ("plays", "is_views_of", "video"): (edges_plays_to_video[:, 0], edges_plays_to_video[:, 1]),

        ("shares", "is_shares_of", "video"): (edges_shares_to_video[:, 0], edges_shares_to_video[:, 1]),

        ("collections", "is_collections_of", "video"): (edges_collections_to_video[:, 0], edges_collections_to_video[:, 1]),

        ("comments", "is_comments_of", "video"): (edges_comments_to_video[:, 0], edges_comments_to_video[:, 1]),

        ("video", "is_history_of", "video"): (edges_video_to_video_history[:, 0], edges_video_to_video_history[:, 1]),

        ("video", "has_same_topic_as", "video"): (edges_video_to_video_topic[:, 0], edges_video_to_video_topic[:, 1]),

        ("video", "has_same_author_as", "video"): (edges_video_to_video_author[:, 0], edges_video_to_video_author[:, 1]),
    })
    # 提取影响力等级并创建一个映射字典
    labels = {int(video_id): video_data["影响力等级"] for video_id, video_data in data.items()}

    #保存图对象以及标签
    graph_data = {
        "graph": g,  # 已经构建好的 DGL 图
        "labels": labels  # 标签字典，存储每个视频节点的 "影响力等级"
    }

    # 保存图和标签为 pickle 文件
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)
else:
    graph_data = pickle.load(open(graph_path, 'rb'))
    g = graph_data['graph']
    labels = graph_data['labels']

def video_feat(url, platform, root_path='./video_feats'):
    
    proot = {'抖音': 'douyin',
             '头条': 'toutiao',
             '西瓜': 'xigua',
             '快手': 'kuaishou',
             '哔哩哔哩': 'bilibili'}
    file_path = find_file(url, platform)
    vid = file_path.split('/')[-1].split('.')[0]
    feat_path = os.path.join(root_path, proot[platform], vid + '.pth')
    if os.path.exists(feat_path):
        try:
            return torch.load(feat_path).float()
        except:
            print(f"Error: {feat_path}")
            return torch.zeros((3584))
    else:
        return torch.zeros((3584))

print("Extracting features...")
#为每个节点添加特征
if 'feat' not in g.nodes['video'].data:
    g.nodes['video'].data['feat'] = torch.stack([video_feat(video['url'], video['平台']) for video in tqdm(video_info)])
    g.nodes['video'].data['train_mask'] = train_mask
    g.nodes['video'].data['test_mask'] = test_mask
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)

if 'feat' not in g.nodes['description'].data:
    g.nodes['description'].data['feat'] = torch.stack([txt2feat(des["text"]) for des in tqdm(des_info)])
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)

if 'feat' not in g.nodes['title'].data:
    g.nodes['title'].data['feat'] = torch.stack([txt2feat(title["text"]) for title in tqdm(title_info)])
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)

if 'feat' not in g.nodes['topic'].data:
    g.nodes['topic'].data['feat'] = torch.stack([txt2feat(topic["text"]) for topic in tqdm(topic_info)])
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)

if 'feat' not in g.nodes['platform'].data:
    g.nodes['platform'].data['feat'] = torch.stack([txt2feat(platform["text"]) for platform in tqdm(platform_info)])
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)

if 'feat' not in g.nodes['comment'].data:
    g.nodes['comment'].data['feat'] = torch.stack([torch.cat((txt2feat(comment["text"]), time2feat(comment["time"]))) for comment in tqdm(comment_info)])
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)

if 'feat' not in g.nodes['time'].data:
    g.nodes['time'].data['feat'] = torch.stack([time2feat(time["text"]) for time in tqdm(time_info)])
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)

if 'feat' not in g.nodes['ctime'].data:
    g.nodes['ctime'].data['feat'] = torch.stack([time2feat(ctime["text"]) for ctime in tqdm(ctime_info)])
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)

g.nodes['video_time'].data['feat'] = torch.tensor([[math.log(float(video_time["text"]) + 1)] for video_time in tqdm(video_time_info)])
g.nodes['fans'].data['feat'] = torch.tensor([[math.log(float(fans["text"]) + 1)] for fans in tqdm(fans_info)])
g.nodes['likes'].data['feat'] = torch.tensor([[math.log(float(likes["text"]) + 1)] for likes in tqdm(likes_info)])
g.nodes['plays'].data['feat'] = torch.tensor([[math.log(float(plays["text"]) + 1)] for plays in tqdm(plays_info)])
g.nodes['shares'].data['feat'] = torch.tensor([[math.log(float(shares["text"]) + 1)] for shares in tqdm(shares_info)])
g.nodes['collections'].data['feat'] = torch.tensor([[math.log(float(collections["text"]) + 1)] for collections in tqdm(collections_info)])
g.nodes['comments'].data['feat'] = torch.tensor([[math.log(float(comment["text"]) + 1)] for comment in tqdm(comments_info)])


#保存图对象以及标签
graph_data = {
    "graph": g,  # 已经构建好的 DGL 图
    "labels": labels  # 标签字典，存储每个视频节点的 "影响力等级"
}

# 保存图和标签为 pickle 文件
with open(graph_path, 'wb') as f:
    pickle.dump(graph_data, f)
print("Graph and labels saved as graph_with_labels.pkl")


# 清除并写入CSV文件的函数，处理特殊字符
def clear_and_write_csv(file_path, data, fieldnames):
    # Function to replace any backslash escape sequences (like \n, \r, \t, etc.) with a space
    def clean_string(value):
        if isinstance(value, str):
            # Replace any backslash escape sequence with a space
            return ''.join(' ' if ch == '\\' else ch for ch in value)
        return value

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()  # Write the header
        for row in data:
            # Clean each value in the row
            row = {key: clean_string(value) for key, value in row.items()}
            writer.writerow(row)  # Write the cleaned row

# 将视频、描述、标题等节点数据存入 CSV
clear_and_write_csv("data/video_info.csv", video_info, ["id", "url", "影响力等级", "平台"])
clear_and_write_csv("data/des_info.csv", des_info, ["id", "text"])
clear_and_write_csv("data/title_info.csv", title_info, ["id", "text"])
clear_and_write_csv("data/topic_info.csv", topic_info, ["id", "text"])
clear_and_write_csv("data/video_time_info.csv", video_time_info, ["id", "text"])
clear_and_write_csv("data/platform_info.csv", platform_info, ["id", "text"])
clear_and_write_csv("data/comment_info.csv", comment_info, ["id", "text", "time"])
clear_and_write_csv("data/time_info.csv", time_info, ["id", "text"])
clear_and_write_csv("data/fans_info.csv", fans_info, ["id", "text"])
clear_and_write_csv("data/likes_info.csv", likes_info, ["id", "text"])
clear_and_write_csv("data/plays_info.csv", plays_info, ["id", "text"])
clear_and_write_csv("data/shares_info.csv", shares_info, ["id", "text"])
clear_and_write_csv("data/collections_info.csv", collections_info, ["id", "text"])
clear_and_write_csv("data/comments_info.csv", comments_info, ["id", "text"])
