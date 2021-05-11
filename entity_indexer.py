#!/usr/bin/env python
import json
from typing import List
import numpy as np


class EntityIndexer(object):
    def __init__(self, entity_file: str,
                 entity_embedding_file=None,
                 unk_token: str = '[UNK_ENT]',
                 pad_token: str = '[PAD_ENT]',
                 nil_token: str = '[NIL_ENT]'):
        self.name_to_entities = {}
        self.id_to_entities = {}
        self.token_ids_to_entities = {}
        self.max_id = -1
        self.is_compact = False
        self.entity_embedding = None

        with open(entity_file, 'r') as reader:
            for line in reader:
                line = json.loads(line.strip())
                entity_name = line["entity_name"]
                entity_id = line["id"]

                assert entity_name not in self.name_to_entities
                assert entity_id not in self.id_to_entities

                self.name_to_entities[entity_name] = line
                self.id_to_entities[entity_id] = line
                self.max_id = max(self.max_id, entity_id)

        if entity_embedding_file is not None:
            
            with open(entity_embedding_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip().split()
                self.entity_embedding_dim = int(first_line[1])
                self.entity_embedding = np.zeros((self.max_id + 1, self.entity_embedding_dim))
                tmp_entity_embedding = []
                entity_id_list = []
                for line in f.readlines():
                    line_split = line.strip().split()
                    entity_name = line_split[0]
                    entity_id = self.convert_name_to_id(entity_name)
                    entity_embedding = [float(x) for x in line_split[1:]]
                    self.entity_embedding[entity_id] = entity_embedding
                    tmp_entity_embedding.append(entity_embedding)
                    entity_id_list.append(entity_id)
            
            # Deal with missing entities
            """
            mu = np.mean(tmp_entity_embedding, axis=0)
            sigma = np.std(tmp_entity_embedding, axis=0)
            for i in range(self.max_id):
                if not i in entity_id_list:
                    self.entity_embedding[i] = sigma * np.random.randn(self.entity_embedding_dim) + mu
            """

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.nil_token = nil_token

        assert pad_token in self.name_to_entities and self.name_to_entities[
            pad_token]["id"] == 0
        assert unk_token in self.name_to_entities
        assert nil_token in self.name_to_entities

        self.unk_id = self.name_to_entities[unk_token]["id"]
        self.is_compact = (len(self.name_to_entities) == self.max_id + 1)

    def convert_name_to_id(self, name: str):
        return self.name_to_entities[name]["id"] if name in self.name_to_entities else self.unk_id

    def convert_id_to_name(self, index: int):
        return self.id_to_entities[index]["entity_name"] if index in self.id_to_entities else self.unk_token

    def convert_names_to_ids(self, names: List[str]):
        return [self.name_to_entities[name]["id"] if name in self.name_to_entities
                else self.unk_id
                for name in names]

    def convert_ids_to_names(self, ids: List[int]):
        return [self.id_to_entities[i]["entity_name"] if i in self.id_to_entities
                else self.unk_token
                for i in ids]

    def num_entities(self):
        return self.max_id + 1
