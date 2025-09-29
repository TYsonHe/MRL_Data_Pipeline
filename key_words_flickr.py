import json
import os
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 首次运行需要下载 nltk 数据（取消注释下面两行如果第一次运行）
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')


def extract_keywords_from_captions(captions, top_k=10):
    stop_words = set(stopwords.words('english'))
    all_words = []

    for caption in captions:
        # 转小写，去除标点
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(caption)
        # 过滤停用词和长度 <= 2 的词
        filtered = [
            word for word in tokens if word not in stop_words and len(word) > 2]
        all_words.extend(filtered)

    # 统计词频并取最常见的 top_k 个
    word_counts = Counter(all_words)
    most_common = [word for word, _ in word_counts.most_common(top_k)]
    # 如果不足 top_k，就返回全部
    # 将他们变成一个string 空格隔开
    return ' '.join(most_common[:top_k])


def convert_split_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {}

    for split in ['train', 'val', 'test']:
        if split not in data:
            continue
        new_data[split] = {}
        for img_name, captions in data[split].items():
            keywords = extract_keywords_from_captions(captions, top_k=10)
            new_data[split][img_name] = keywords

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 已保存新文件到: {output_path}")


if __name__ == "__main__":
    input_file = "E:\Projects_draft\MRL_Data_Pipeline\Datasets\Flickr30k\\flickr30k_processed\splits.json"
    output_file = "E:\Projects_draft\MRL_Data_Pipeline\Datasets\Flickr30k\\flickr30k_processed\split_keywords.json"

    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件: {input_file}")
    else:
        convert_split_json(input_file, output_file)
