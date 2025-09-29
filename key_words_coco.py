import json
import os
import string
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 首次运行请取消注释以下两行
# nltk.download('punkt')
# nltk.download('stopwords')


def extract_keywords_from_captions(captions, top_k=10):
    stop_words = set(stopwords.words('english'))
    all_words = []

    for caption in captions:
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(caption)
        filtered = [
            word for word in tokens if word not in stop_words and len(word) > 2]
        all_words.extend(filtered)

    word_counts = Counter(all_words)
    most_common = [word for word, _ in word_counts.most_common(top_k)]
    return ' '.join(most_common[:top_k])  # 返回空格分隔的字符串


if __name__ == "__main__":
    annotationsFile = './Datasets/MS_COCO/annotations/captions_val2014.json'
    outputAnnotationsFile = './Datasets/MS_COCO/annotations/captions_val2014_keywords.json'

    print("Loading annotations file...")
    with open(annotationsFile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Annotations file loaded.")

    # Step 1: 按 image_id 聚合所有 caption
    captions_by_image = defaultdict(list)
    for ann in data['annotations']:
        captions_by_image[ann['image_id']].append(ann['caption'])

    # Step 2: 为每张图生成关键词 caption
    new_annotations = []
    new_annotation_id = 1  # COCO 要求 annotation id 唯一，我们重新编号

    for image_id, captions in captions_by_image.items():
        keywords_str = extract_keywords_from_captions(captions, top_k=10)
        new_ann = {
            "image_id": image_id,
            "id": new_annotation_id,
            "caption": keywords_str
        }
        new_annotations.append(new_ann)
        new_annotation_id += 1

    # Step 3: 替换原 data 中的 annotations
    data['annotations'] = new_annotations

    # （可选）验证：确保 images 列表中的 image_id 与 annotations 一致
    # 这里我们保留所有原 images，因为 COCO 需要完整 image 列表

    print(f"Total images processed: {len(captions_by_image)}")
    print(f"New annotations count: {len(new_annotations)}")

    # Step 4: 保存
    print("Writing modified annotations to file...")
    with open(outputAnnotationsFile, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)

    print("✅ Finished. Output saved to:", outputAnnotationsFile)
