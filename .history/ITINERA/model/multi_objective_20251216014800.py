"""
多目标优化模块（NSGA-III + 两阶段优化）
基于帕累托前沿的地块推荐

两阶段优化流程：
1. 阶段1：硬约束过滤 + 帕累托前沿筛选（粗筛）
2. 阶段2：LLM权重加权排序（精排）
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re


class HardConstraintParser:
    """
    硬约束解析器
    将LLM提取的文本约束转换为可执行的过滤条件
    """
    
    # 数值提取正则
    NUMBER_PATTERN = re.compile(r'[≥≤>=<]?\s*(\d+(?:\.\d+)?)')
    
    # 约束类型到字段的映射
    CONSTRAINT_FIELD_MAP = {
        '区域': '宗地坐落',
        '用地类型': '土地用途',
        '面积': '宗地面积(平方米)',
        '成本': '挂牌起始价(万元)',
        '单价': '价格_万元/㎡',
        '交通': '交通_便利评分(0-10)',
        '地铁': '交通_地铁最近距离(m)',
    }
    
    # 面积单位转换
    AREA_UNITS = {
        '亩': 666.67,
        '公顷': 10000,
        '平方米': 1,
        '㎡': 1,
        '平米': 1,
    }
