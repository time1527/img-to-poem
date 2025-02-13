import re
import os
import sys
import json
import glob
import zhon.hanzi

repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_path)
from settings import WORKS


def get_works():
    json_files = glob.glob(os.path.join(WORKS, "*.json"))

    ct_works = []
    c_works = []

    for json_file in json_files:
        works = json.load(open(json_file))
        for item in works:
            k = item["Kind"]
            c = item["Content"]
            t = item["Translation"]
            if k not in ["诗", "词", "曲"]:
                continue
            if c == None or len(c) <= 1 or c.find("【") != -1:
                continue
            c_list = c.split("\r\n")

            if t == None or len(t.strip()) == 0:
                t_list = []
            else:
                t_list = t.split("\r\n")

            # 根据\r\n可切分：ct
            if len(c_list) == len(t_list):
                ct_works.extend(
                    [
                        {
                            "content": x.strip().replace("“", "").replace("”", ""),
                            "translation": y.strip().replace("“", "").replace("”", ""),
                            # "author":item["Author"],
                            # "title":item['Title']
                        }
                        for x, y in zip(c_list, t_list)
                    ]
                )
            # 直接切分
            new_c = c.replace("\r\n", "").replace("\n", "").replace(" ", "")
            new_c_list = re.findall(zhon.hanzi.sentence, new_c)
            c_dict_list = list(
                map(
                    lambda x: {
                        "content": x.strip().replace("“", "").replace("”", ""),
                        "author": item["Author"],
                        # "title":item['Title']
                    },
                    new_c_list,
                )
            )
            c_works.extend(c_dict_list)
    with open(os.path.join(WORKS, "ct_works.jsonl"), "w", encoding="utf-8") as f:
        for item in ct_works:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(os.path.join(WORKS, "c_works.jsonl"), "w", encoding="utf-8") as f:
        for item in c_works:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    author = "李白 杜甫 苏轼 王维 杜牧 陆游 李煜 元稹 韩愈 岑参 齐己 贾岛 柳永 曹操 李贺 曹植 张籍 孟郊 皎然 许浑 罗隐 贯休 韦庄 屈原 王勃 张祜 王建 晏殊 岳飞 姚合 卢纶 秦观 钱起 朱熹 韩偓 高适 方干 李峤 赵嘏 贺铸 郑谷 郑燮 张说 张炎 白居易 辛弃疾 李清照 刘禹锡 李商隐 陶渊明 孟浩然 柳宗元 王安石 欧阳修 韦应物 温庭筠 刘长卿 王昌龄 杨万里 诸葛亮 范仲淹 陆龟蒙 晏几道 周邦彦 杜荀鹤 吴文英 马致远 皮日休 左丘明 张九龄 权德舆 黄庭坚 司马迁 皇甫冉 卓文君 文天祥 刘辰翁 陈子昂 纳兰性德"
    author_list = author.split(" ")
    select_c_works = [item for item in c_works if item["author"] in author_list]
    with open(os.path.join(WORKS, "select_c_works.jsonl"), "w", encoding="utf-8") as f:
        for item in select_c_works:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    get_works()
