"""DocRED: A Large-Scale Document-Level Relation Extraction Dataset"""


import json

import datasets


_CITATION = """\
@inproceedings{yao-etal-2019-docred,
    title = "{D}oc{RED}: A Large-Scale Document-Level Relation Extraction Dataset",
    author = "Yao, Yuan  and
      Ye, Deming  and
      Li, Peng  and
      Han, Xu  and
      Lin, Yankai  and
      Liu, Zhenghao  and
      Liu, Zhiyuan  and
      Huang, Lixin  and
      Zhou, Jie  and
      Sun, Maosong",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1074",
    doi = "10.18653/v1/P19-1074",
    pages = "764--777",
}
"""

_DESCRIPTION = """\
Multiple entities in a document generally exhibit complex inter-sentence relations, and cannot be well handled by \
existing relation extraction (RE) methods that typically focus on extracting intra-sentence relations for single \
entity pairs. In order to accelerate the research on document-level RE, we introduce DocRED, a new dataset constructed \
from Wikipedia and Wikidata with three features:
    - DocRED annotates both named entities and relations, and is the largest human-annotated dataset for document-level RE from plain text.
    - DocRED requires reading multiple sentences in a document to extract entities and infer their relations by synthesizing all information of the document.
    - Along with the human-annotated data, we also offer large-scale distantly supervised data, which enables DocRED to be adopted for both supervised and weakly supervised scenarios.
"""

_URLS = {
    "dev": "data/dev.json.gz",
    "train_distant": "data/train_distant.json.gz",
    "train_annotated": "data/train_annotated.json.gz",
    "test": "data/test.json.gz",
    "rel_info": "data/rel_info.json.gz",
}


class DocRed(datasets.GeneratorBasedBuilder):
    """DocRED: A Large-Scale Document-Level Relation Extraction Dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "title": datasets.Value("string"),
                    "sents": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                    "vertexSet": [
                        [
                            {
                                "name": datasets.Value("string"),
                                "sent_id": datasets.Value("int32"),
                                "pos": datasets.features.Sequence(datasets.Value("int32")),
                                "type": datasets.Value("string"),
                            }
                        ]
                    ],
                    "labels": datasets.features.Sequence(
                        {
                            "head": datasets.Value("int32"),
                            "tail": datasets.Value("int32"),
                            "relation_id": datasets.Value("string"),
                            "relation_text": datasets.Value("string"),
                            "evidence": datasets.features.Sequence(datasets.Value("int32")),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/thunlp/DocRED",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloads = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloads["dev"], "rel_info": downloads["rel_info"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": downloads["test"], "rel_info": downloads["rel_info"]}
            ),
            datasets.SplitGenerator(
                name="train_annotated",
                gen_kwargs={"filepath": downloads["train_annotated"], "rel_info": downloads["rel_info"]},
            ),
            datasets.SplitGenerator(
                name="train_distant",
                gen_kwargs={"filepath": downloads["train_distant"], "rel_info": downloads["rel_info"]},
            ),
        ]

    def _generate_examples(self, filepath, rel_info):
        """Generate DocRED examples."""

        with open(rel_info, encoding="utf-8") as f:
            relation_name_map = json.load(f)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        for idx, example in enumerate(data):

            # Test set has no labels - Results need to be uploaded to Codalab
            if "labels" not in example.keys():
                example["labels"] = []

            for label in example["labels"]:
                # Rename and include full relation names
                label["relation_text"] = relation_name_map[label["r"]]
                label["relation_id"] = label["r"]
                label["head"] = label["h"]
                label["tail"] = label["t"]
                del label["r"]
                del label["h"]
                del label["t"]

            yield idx, example
