#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common definitions for NER
"""

from util import one_hot
#'B-0','I-0','B-1','I-1','B-2','I-2','B-3','I-3','B-4','I-4','O'
LBLS = [
    'B-e','I-e','O'
    ]
NONE = "O"
LMAP = {k: one_hot(len(LBLS),i) for i, k in enumerate(LBLS)}
UNK = "UUUNKKK"

EMBED_SIZE = 60
