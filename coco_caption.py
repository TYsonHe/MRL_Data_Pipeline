import json
import random
import re
from pathlib import Path
import argparse
from typing import List, Dict, Any
import nltk
from nltk.corpus import wordnet
import string

# 下载必要的NLTK数据
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


class CaptionProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        """
        初始化Caption处理器

        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 常用同义词词典（扩展版）
        self.synonym_dict = {
            'a': ['an', 'one', 'single'],
            'the': ['this', 'that', 'these', 'those'],
            'is': ['are', 'was', 'were', 'be', 'being', 'has been'],
            'are': ['is', 'were', 'was', 'be', 'being', 'have been'],
            'and': ['plus', 'with', 'along with', 'as well as'],
            'in': ['inside', 'within', 'into', 'at'],
            'on': ['upon', 'atop', 'above', 'over'],
            'with': ['along with', 'together with', 'accompanied by'],
            'for': ['for the purpose of', 'intended for', 'aimed at'],
            'of': ['belonging to', 'from', 'related to'],
            'to': ['toward', 'towards', 'into'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'petite'],
            'good': ['great', 'excellent', 'wonderful', 'fantastic', 'amazing'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'dreadful'],
            'beautiful': ['gorgeous', 'stunning', 'attractive', 'pretty', 'lovely'],
            'ugly': ['unattractive', 'hideous', 'unsightly', 'unpleasant'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried'],
            'old': ['ancient', 'aged', 'elderly', 'mature', 'vintage'],
            'new': ['fresh', 'recent', 'modern', 'contemporary', 'latest'],
            'red': ['crimson', 'scarlet', 'ruby', 'cherry', 'burgundy'],
            'blue': ['azure', 'navy', 'sky-blue', 'cerulean', 'cobalt'],
            'green': ['emerald', 'lime', 'forest', 'olive', 'mint'],
            'yellow': ['golden', 'amber', 'lemon', 'mustard', 'saffron'],
            'black': ['dark', 'ebony', 'jet-black', 'charcoal', 'obsidian'],
            'white': ['snowy', 'ivory', 'cream', 'pearl', 'milky'],
            'person': ['individual', 'human', 'people', 'man', 'woman'],
            'people': ['individuals', 'persons', 'humans', 'crowd'],
            'man': ['male', 'gentleman', 'guy', 'person'],
            'woman': ['female', 'lady', 'girl', 'person'],
            'child': ['kid', 'youngster', 'youth', 'minor'],
            'dog': ['canine', 'puppy', 'hound', 'mutt'],
            'cat': ['feline', 'kitten', 'tomcat', 'pussycat'],
            'car': ['vehicle', 'automobile', 'auto', 'motorcar'],
            'house': ['home', 'residence', 'dwelling', 'building'],
            'tree': ['plant', 'wood', 'forest', 'timber'],
            'food': ['meal', 'dish', 'cuisine', 'edible'],
            'water': ['liquid', 'H2O', 'aqua', 'fluid'],
            'book': ['volume', 'tome', 'publication', 'literature'],
            'phone': ['telephone', 'mobile', 'cellphone', 'device'],
            'computer': ['PC', 'laptop', 'device', 'machine'],
            'table': ['desk', 'surface', 'counter', 'board'],
            'chair': ['seat', 'stool', 'bench', 'armchair'],
            'bed': ['mattress', 'cot', 'bunk', 'futon'],
            'door': ['entrance', 'gate', 'entry', 'portal'],
            'window': ['pane', 'glass', 'casement', 'sash'],
            'street': ['road', 'avenue', 'boulevard', 'lane'],
            'building': ['structure', 'edifice', 'construction', 'facility'],
            'room': ['space', 'chamber', 'area', 'compartment'],
            'kitchen': ['cookhouse', 'culinary', 'galley', 'pantry'],
            'bathroom': ['washroom', 'restroom', 'lavatory', 'toilet'],
            'living': ['family', 'sitting', 'lounge', 'reception'],
            'dining': ['eating', 'meal', 'restaurant', 'canteen'],
            'bedroom': ['sleeping', 'chamber', 'rest', 'personal'],
            'office': ['workplace', 'study', 'professional', 'business'],
            'park': ['garden', 'recreation', 'playground', 'square'],
            'beach': ['shore', 'coast', 'sand', 'seaside'],
            'mountain': ['hill', 'peak', 'summit', 'range'],
            'river': ['stream', 'creek', 'waterway', 'brook'],
            'lake': ['pond', 'reservoir', 'lagoon', 'pool'],
            'city': ['town', 'urban', 'metropolitan', 'downtown'],
            'country': ['rural', 'countryside', 'village', 'farmland'],
            'outdoor': ['outside', 'exterior', 'open-air', 'al fresco'],
            'indoor': ['inside', 'interior', 'enclosed', 'covered'],
            'day': ['daytime', 'daylight', 'morning', 'afternoon'],
            'night': ['evening', 'darkness', 'nighttime', 'sunset'],
            'morning': ['dawn', 'sunrise', 'AM', 'early'],
            'evening': ['dusk', 'sunset', 'PM', 'late'],
            'summer': ['warm', 'hot', 'sunny', 'bright'],
            'winter': ['cold', 'snowy', 'icy', 'freezing'],
            'spring': ['bloom', 'fresh', 'new', 'growing'],
            'autumn': ['fall', 'harvest', 'colorful', 'changing'],
            'sunny': ['bright', 'clear', 'warm', 'radiant'],
            'cloudy': ['overcast', 'gray', 'dull', 'gloomy'],
            'rainy': ['wet', 'showery', 'drizzly', 'stormy'],
            'snowy': ['white', 'icy', 'frosty', 'wintry'],
            'windy': ['breezy', 'blustery', 'gusty', 'drafty'],
            'hot': ['warm', 'heated', 'scorching', 'boiling'],
            'cold': ['cool', 'chilly', 'freezing', 'icy'],
            'warm': ['hot', 'heated', 'comfortable', 'mild'],
            'cool': ['cold', 'chilly', 'refreshing', 'mild'],
            'happy': ['joyful', 'cheerful', 'glad', 'pleased'],
            'sad': ['unhappy', 'sorrowful', 'depressed', 'gloomy'],
            'angry': ['mad', 'furious', 'irritated', 'annoyed'],
            'excited': ['thrilled', 'enthusiastic', 'eager', 'stimulated'],
            'calm': ['peaceful', 'relaxed', 'serene', 'tranquil'],
            'busy': ['active', 'occupied', 'engaged', 'working'],
            'quiet': ['silent', 'still', 'peaceful', 'hushed'],
            'loud': ['noisy', 'boisterous', 'vocal', 'audible'],
            'soft': ['gentle', 'smooth', 'delicate', 'tender'],
            'hard': ['firm', 'solid', 'stiff', 'rigid'],
            'light': ['bright', 'luminous', 'illuminated', 'shiny'],
            'dark': ['dim', 'shadowy', 'black', 'gloomy'],
            'clean': ['tidy', 'neat', 'spotless', 'pure'],
            'dirty': ['messy', 'filthy', 'unclean', 'stained'],
            'empty': ['vacant', 'bare', 'hollow', 'unoccupied'],
            'full': ['complete', 'filled', 'packed', 'loaded'],
            'open': ['unlocked', 'available', 'accessible', 'exposed'],
            'closed': ['shut', 'locked', 'sealed', 'covered'],
            'young': ['youthful', 'junior', 'new', 'fresh'],
            'adult': ['mature', 'grown-up', 'elder', 'senior'],
            'family': ['relatives', 'kin', 'household', 'clan'],
            'friend': ['companion', 'buddy', 'pal', 'associate'],
            'group': ['team', 'crowd', 'collection', 'assembly'],
            'alone': ['single', 'solitary', 'individual', 'separate'],
            'together': ['combined', 'united', 'joined', 'collective'],
            'playing': ['gaming', 'sporting', 'recreating', 'amusing'],
            'working': ['laboring', 'employed', 'occupied', 'functioning'],
            'sitting': ['seated', 'resting', 'perched', 'settled'],
            'standing': ['upright', 'erect', 'vertical', 'raised'],
            'walking': ['strolling', 'hiking', 'marching', 'trekking'],
            'running': ['jogging', 'sprinting', 'dashing', 'rushing'],
            'eating': ['dining', 'consuming', 'feasting', 'nibbling'],
            'drinking': ['sipping', 'gulping', 'swallowing', 'imbibing'],
            'talking': ['speaking', 'chatting', 'conversing', 'discussing'],
            'sleeping': ['resting', 'napping', 'dozing', 'slumbering'],
            'reading': ['studying', 'perusing', 'scanning', 'browsing'],
            'writing': ['composing', 'scribbling', 'typing', 'recording'],
            'drawing': ['sketching', 'painting', 'illustrating', 'creating'],
            'cooking': ['preparing', 'baking', 'frying', 'grilling'],
            'cleaning': ['washing', 'tidying', 'scrubbing', 'wiping'],
            'shopping': ['buying', 'purchasing', 'browsing', 'acquiring'],
            'driving': ['operating', 'steering', 'cruising', 'traveling'],
            'flying': ['soaring', 'gliding', 'aviating', 'floating'],
            'swimming': ['diving', 'splashing', 'bathing', 'paddling'],
        }

        # 掩码符号
        self.mask_tokens = ['<mask>', '</think>', '___', '???', '***']

    def get_synonyms(self, word: str) -> List[str]:
        """
        获取单词的同义词

        Args:
            word: 输入单词

        Returns:
            同义词列表
        """
        # 首先检查预定义词典
        word_lower = word.lower()
        if word_lower in self.synonym_dict:
            return self.synonym_dict[word_lower]

        # 使用WordNet获取同义词
        synonyms = []
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word_lower and synonym not in synonyms:
                        synonyms.append(synonym)
                        if len(synonyms) >= 3:  # 限制同义词数量
                            break
                if len(synonyms) >= 3:
                    break
        except:
            pass

        return synonyms[:3] if synonyms else []

    def synonym_replacement(self, text: str, replacement_prob: float = 0.3) -> str:
        """
        同义词替换

        Args:
            text: 输入文本
            replacement_prob: 替换概率

        Returns:
            替换后的文本
        """
        words = text.split()
        new_words = words.copy()

        for i, word in enumerate(words):
            # 跳过标点符号
            if word in string.punctuation:
                continue

            # 随机决定是否替换
            if random.random() < replacement_prob:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    new_words[i] = random.choice(synonyms)

        return ' '.join(new_words)

    def random_masking(self, text: str, mask_prob: float = 0.15) -> str:
        """
        随机掩词

        Args:
            text: 输入文本
            mask_prob: 掩码概率

        Returns:
            掩码后的文本
        """
        words = text.split()
        new_words = words.copy()

        for i, word in enumerate(words):
            # 跳过标点符号和短词
            if word in string.punctuation or len(word) <= 2:
                continue

            # 随机决定是否掩码
            if random.random() < mask_prob:
                mask_token = random.choice(self.mask_tokens)
                new_words[i] = mask_token

        return ' '.join(new_words)

    def process_caption_entry_synonym(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个caption条目 - 只做同义改写

        Args:
            entry: caption条目

        Returns:
            处理后的caption条目
        """
        new_entry = entry.copy()

        # 对caption文本进行同义改写
        if 'caption' in new_entry:
            original_caption = new_entry['caption']
            perturbed_caption = self.synonym_replacement(original_caption)
            new_entry['caption'] = perturbed_caption

            # 添加扰动标记
            new_entry['perturbed'] = perturbed_caption != original_caption
            new_entry['perturbation_type'] = 'synonym_replacement'

        return new_entry

    def process_caption_entry_mask(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个caption条目 - 只做随机掩词

        Args:
            entry: caption条目

        Returns:
            处理后的caption条目
        """
        new_entry = entry.copy()

        # 对caption文本进行随机掩词
        if 'caption' in new_entry:
            original_caption = new_entry['caption']
            perturbed_caption = self.random_masking(original_caption)
            new_entry['caption'] = perturbed_caption

            # 添加扰动标记
            new_entry['perturbed'] = perturbed_caption != original_caption
            new_entry['perturbation_type'] = 'random_masking'

        return new_entry

    def process_caption_entry_both(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个caption条目 - 同义改写+随机掩词

        Args:
            entry: caption条目

        Returns:
            处理后的caption条目
        """
        new_entry = entry.copy()

        # 对caption文本进行两种扰动
        if 'caption' in new_entry:
            original_caption = new_entry['caption']

            # 先进行同义改写
            synonym_caption = self.synonym_replacement(original_caption)
            # 再进行随机掩词
            perturbed_caption = self.random_masking(synonym_caption)

            new_entry['caption'] = perturbed_caption

            # 添加扰动标记
            new_entry['perturbed'] = perturbed_caption != original_caption
            new_entry['perturbation_type'] = 'synonym_and_mask'

        return new_entry

    def process_captions_file(self, input_file: str, output_file: str, processor_func, perturbation_name: str):
        """
        处理captions JSON文件

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            processor_func: 处理函数
            perturbation_name: 扰动名称
        """
        print(f"正在处理文件: {input_file} -> {perturbation_name}")

        # 读取原始文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"原始数据包含 {len(data.get('images', []))} 张图像")
        print(f"原始数据包含 {len(data.get('annotations', []))} 个标注")

        # 处理图像信息
        if 'images' in data:
            processed_images = data['images'].copy()
            data['images'] = processed_images

        # 处理标注信息
        if 'annotations' in data:
            processed_annotations = []
            perturbation_count = 0

            for annotation in data['annotations']:
                processed_annotation = processor_func(annotation)
                processed_annotations.append(processed_annotation)

                if processed_annotation.get('perturbed', False):
                    perturbation_count += 1

            data['annotations'] = processed_annotations

            print(
                f"扰动处理完成: {perturbation_count}/{len(processed_annotations)} 个标注被扰动")
            print(
                f"扰动比例: {perturbation_count/len(processed_annotations)*100:.2f}%")

        # 添加处理元信息
        data['processing_info'] = {
            'processing_date': '2025-11-24',
            'perturbation_method': perturbation_name,
            'description': f'文本扰动处理: {perturbation_name}'
        }

        # 保存处理后的文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"处理完成，输出文件: {output_file}")
        print("-" * 50)

    def process_all_files(self):
        """
        处理所有captions文件，生成3个版本
        """
        files_to_process = [
            'captions_train2014.json',
            'captions_val2014.json'
        ]

        for filename in files_to_process:
            input_path = self.input_dir / filename

            if not input_path.exists():
                print(f"警告: 文件不存在 {input_path}")
                continue

            # 生成3个版本的输出文件
            base_name = filename.replace('.json', '')

            # 版本1: 只做同义改写
            output_synonym = self.output_dir / f"{base_name}_synonym.json"
            self.process_captions_file(
                str(input_path),
                str(output_synonym),
                self.process_caption_entry_synonym,
                "synonym_replacement"
            )

            # 版本2: 只做随机掩词
            output_mask = self.output_dir / f"{base_name}_mask.json"
            self.process_captions_file(
                str(input_path),
                str(output_mask),
                self.process_caption_entry_mask,
                "random_masking"
            )

            # 版本3: 同义改写+随机掩词
            output_both = self.output_dir / f"{base_name}_both.json"
            self.process_captions_file(
                str(input_path),
                str(output_both),
                self.process_caption_entry_both,
                "synonym_and_mask"
            )


def main():
    parser = argparse.ArgumentParser(
        description='MSCOCO Captions文本扰动处理工具 - 3个版本')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录路径（包含captions_train2014.json和captions_val2014.json）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径')

    args = parser.parse_args()

    # 创建处理器
    processor = CaptionProcessor(args.input_dir, args.output_dir)

    # 处理所有文件
    processor.process_all_files()

    print("\n所有caption文件处理完成！")
    print(f"输出目录: {args.output_dir}")
    print("\n生成的文件:")
    print("├── captions_train2014_synonym.json    # 只做同义改写")
    print("├── captions_train2014_mask.json       # 只做随机掩词")
    print("├── captions_train2014_both.json       # 同义改写+随机掩词")
    print("├── captions_val2014_synonym.json      # 只做同义改写")
    print("├── captions_val2014_mask.json         # 只做随机掩词")
    print("└── captions_val2014_both.json          # 同义改写+随机掩词")


if __name__ == "__main__":
    main()
