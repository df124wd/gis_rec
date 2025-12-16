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

    @classmethod
    def parse_constraints(cls, hard_constraints: List[Dict], proxy=None) -> List[Dict]:
        """
        将文本约束解析为可执行的过滤条件
        
        Args:
            hard_constraints: [{text, type, is_negative}, ...]
            proxy: LLM代理（用于复杂约束解析）
        
        Returns:
            executable_constraints: [{field, operator, value, is_negative}, ...]
        """
        executable = []
        
        for c in hard_constraints:
            text = c.get('text', '').strip()
            ctype = c.get('type', '')
            is_neg = c.get('is_negative', False)
            
            if not text:
                continue
            
            parsed = cls._parse_single_constraint(text, ctype, is_neg, proxy)
            if parsed:
                executable.append(parsed)
        
        return executable
    
    @classmethod
    def _parse_single_constraint(cls, text: str, ctype: str, is_neg: bool, proxy=None) -> Optional[Dict]:
        """解析单个约束"""
        
        # 1. 区域约束（包含匹配）
        if ctype == '区域':
            # 提取区域名称
            districts = ['天河区', '越秀区', '海珠区', '荔湾区', '黄埔区', 
                        '白云区', '番禺区', '花都区', '南沙区', '增城区', '从化区']
            for d in districts:
                if d in text or d.replace('区', '') in text:
                    return {
                        'field': '宗地坐落',
                        'operator': 'not_contains' if is_neg else 'contains',
                        'value': d.replace('区', ''),  # 匹配"花都"而非"花都区"更宽松
                        'is_negative': is_neg,
                        'original_text': text
                    }
            # 如果没匹配到具体区，用原文本
            return {
                'field': '宗地坐落',
                'operator': 'not_contains' if is_neg else 'contains',
                'value': text,
                'is_negative': is_neg,
                'original_text': text
            }
        
        # 2. 用地类型约束
        if ctype == '用地类型':
            land_types = ['工业', '商业', '居住', '仓储', '物流', '办公']
            for lt in land_types:
                if lt in text:
                    return {
                        'field': '土地用途',
                        'operator': 'not_contains' if is_neg else 'contains',
                        'value': lt,
                        'is_negative': is_neg,
                        'original_text': text
                    }
            return {
                'field': '土地用途',
                'operator': 'not_contains' if is_neg else 'contains',
                'value': text,
                'is_negative': is_neg,
                'original_text': text
            }

        # 3. 面积约束
        if ctype == '面积':
            # 提取数值和单位
            number_match = cls.NUMBER_PATTERN.search(text)
            if number_match:
                value = float(number_match.group(1))
                # 检测单位并转换为平方米
                for unit, factor in cls.AREA_UNITS.items():
                    if unit in text:
                        value *= factor
                        break
                # 检测操作符
                if '≥' in text or '>=' in text or '至少' in text or '不少于' in text or '以上' in text:
                    op = '>='
                elif '≤' in text or '<=' in text or '不超过' in text or '以下' in text or '最多' in text:
                    op = '<='
                else:
                    op = '>='  # 默认"至少"
                
                return {
                    'field': '宗地面积(平方米)',
                    'operator': op,
                    'value': value,
                    'is_negative': is_neg,
                    'original_text': text
                }
        
        # 4. 成本/价格约束
        if ctype == '成本':
            number_match = cls.NUMBER_PATTERN.search(text)
            if number_match:
                value = float(number_match.group(1))
                # 判断是总价还是单价
                if '元/㎡' in text or '元/平' in text or '单价' in text:
                    field = '价格_万元/㎡'
                    value = value / 10000  # 转换为万元/㎡
                else:
                    field = '挂牌起始价(万元)'
                
                # 检测操作符
                if '≤' in text or '<=' in text or '不超过' in text or '以下' in text:
                    op = '<='
                elif '≥' in text or '>=' in text or '至少' in text or '以上' in text:
                    op = '>='
                else:
                    op = '<='  # 价格默认"不超过"
                
                return {
                    'field': field,
                    'operator': op,
                    'value': value,
                    'is_negative': is_neg,
                    'original_text': text
                }
        
        # 5. 配套约束（地铁距离等）
        if ctype == '配套':
            if '地铁' in text:
                number_match = cls.NUMBER_PATTERN.search(text)
                if number_match:
                    value = float(number_match.group(1))
                    return {
                        'field': '交通_地铁最近距离(m)',
                        'operator': '<=',
                        'value': value,
                        'is_negative': is_neg,
                        'original_text': text
                    }
                else:
                    # 没有具体数值，用默认1500m
                    return {
                        'field': '交通_地铁最近距离(m)',
                        'operator': '<=',
                        'value': 1500,
                        'is_negative': is_neg,
                        'original_text': text
                    }
        
        # 无法解析的约束，返回None（后续用语义匹配兜底）
        return None
