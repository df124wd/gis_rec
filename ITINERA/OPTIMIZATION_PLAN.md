# 优化实施计划

## 目标

将当前的**过度工程化**系统简化为**清晰高效**的推荐引擎，同时增强核心功能。

---

## Phase 1: 清理冗余（1-2天）

### 任务清单

- [ ] **移除结构化约束过滤** (`apply_struct_filters`)
  - 文件：`site_selector.py`
  - 行数：约200行
  - 原因：与语义检索重叠，LLM映射不稳定
  - 影响：无，功能已被语义检索覆盖

- [ ] **移除SAFE融合** (`blend_with_safe`, `init_safe_inference`)
  - 文件：`site_selector.py`
  - 行数：约150行
  - 原因：已禁用，效果不明显
  - 影响：无，已默认关闭

- [ ] **移除TSP访问顺序** (`generate_site_order`)
  - 文件：`site_selector.py`, `spatial.py`
  - 行数：约100行
  - 原因：选址不需要访问顺序
  - 影响：无，与业务场景不符

- [ ] **简化需求覆盖** (`apply_request_overrides`)
  - 保留：工业用途过滤
  - 移除：交通优先重排（已被综合评分覆盖）
  - 行数：减少约50行

### 预期效果

- ✅ 代码量减少：~500行（30%）
- ✅ 响应速度提升：~20%
- ✅ 维护成本降低：~50%

---

## Phase 2: 简化评分（2-3天）

### 当前问题

```python
# 复杂的多层评分
text_score = embedding_similarity()
safe_score = safe_prediction()  # 已禁用
struct_score = rule_satisfaction()
poi_score = composite_score()
final = w_text * text_norm + w_safe * safe + w_struct * struct + w_poi * poi
```

### 优化方案

```python
# 简化为直接加权
def calculate_final_score(site_id, weights):
    """
    统一评分函数
    
    Args:
        site_id: 地块ID
        weights: {'text': 0.3, 'traffic': 0.3, 'price': 0.2, 'region': 0.2}
    
    Returns:
        final_score: 1-10分
        breakdown: {'text': 8.5, 'traffic': 9.0, 'price': 7.5, 'region': 8.0}
    """
    row = self.site_data.loc[site_id]
    
    scores = {
        'text': self.text_similarity_score(site_id),      # 语义相似度
        'traffic': self.traffic_score(row),               # 交通便利性
        'price': self.price_score(row),                   # 性价比
        'region': self.region_score(row)                  # 地区评分
    }
    
    final = sum(weights[k] * scores[k] for k in weights)
    
    return final, scores
```

### 实施步骤

1. 创建新的 `calculate_final_score()` 方法
2. 替换所有评分调用
3. 移除旧的 `composite_score()`, `blend_with_safe()` 等
4. 更新前端展示逻辑

### 预期效果

- ✅ 逻辑清晰，易于理解
- ✅ 分数可解释性提升
- ✅ 便于调试和优化

---

## Phase 3: 增强权重推导（3-4天）

### 当前问题

```python
# 硬编码规则，覆盖有限
if "交通便利" in text:
    return {'traffic': 0.5, 'price': 0.25, 'region': 0.25}
```

### 优化方案

```python
def derive_weights_with_llm(self, user_reqs):
    """
    使用LLM智能推导权重
    
    示例：
    输入："天河区工业用地，交通便利，价格便宜"
    输出：{'traffic': 0.4, 'price': 0.35, 'region': 0.15, 'area': 0.1}
    """
    prompt = f"""
你是专业的选址顾问。请分析用户需求并推导评分权重。

用户需求：{user_reqs}

评分维度：
1. text - 语义相似度（与需求描述的匹配程度）
2. traffic - 交通便利性（地铁/公交/火车距离）
3. price - 性价比（单价越低越好）
4. region - 地区位置（行政区评分）

请根据用户需求的侧重点，分配权重（总和为1）。

返回JSON格式：
{{
    "weights": {{
        "text": 0.3,
        "traffic": 0.4,
        "price": 0.2,
        "region": 0.1
    }},
    "reasoning": "用户明确强调交通便利和价格，因此交通和价格权重较高"
}}
"""
    
    response = self.proxy.chat(
        messages=[{"role": "user", "content": prompt}],
        model=None,
        temperature=0
    )
    
    try:
        result = json.loads(response)
        weights = result['weights']
        reasoning = result.get('reasoning', '')
        
        # 归一化权重
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        print(f"[权重推导] {reasoning}")
        print(f"[权重分配] {weights}")
        
        return weights
    except Exception as e:
        print(f"[权重推导失败] 使用默认权重: {e}")
        return {'text': 0.3, 'traffic': 0.3, 'price': 0.2, 'region': 0.2}
```

### 实施步骤

1. 创建 `derive_weights_with_llm()` 方法
2. 在 `solve()` 中调用
3. 添加权重缓存（避免重复调用）
4. 添加降级策略（LLM失败时使用默认权重）

### 预期效果

- ✅ 权重更贴合用户意图
- ✅ 减少硬编码规则
- ✅ 提升推荐准确性

---

## Phase 4: 添加多目标优化（4-5天）

### 当前问题

```python
# 简单排序，无法处理冲突目标
sorted_results = sorted(candidates, key=lambda x: x.score, reverse=True)
```

### 优化方案

```python
def pareto_optimization(self, candidates, objectives):
    """
    帕累托多目标优化
    
    Args:
        candidates: 候选地块列表
        objectives: [
            {'name': 'traffic', 'maximize': True},
            {'name': 'price', 'maximize': False},  # 价格越低越好
            {'name': 'region', 'maximize': True}
        ]
    
    Returns:
        pareto_front: 帕累托前沿（非支配解集）
    """
    def dominates(a, b, objectives):
        """判断a是否支配b"""
        better_in_any = False
        for obj in objectives:
            name = obj['name']
            maximize = obj['maximize']
            
            a_val = a.scores[name]
            b_val = b.scores[name]
            
            if maximize:
                if a_val < b_val:
                    return False
                if a_val > b_val:
                    better_in_any = True
            else:
                if a_val > b_val:
                    return False
                if a_val < b_val:
                    better_in_any = True
        
        return better_in_any
    
    # 计算帕累托前沿
    pareto_front = []
    for c1 in candidates:
        dominated = False
        for c2 in candidates:
            if c1 != c2 and dominates(c2, c1, objectives):
                dominated = True
                break
        if not dominated:
            pareto_front.append(c1)
    
    return pareto_front
```

### 使用示例

```python
# 在solve()中使用
candidates = self.get_candidate_sites()

# 定义优化目标
objectives = [
    {'name': 'traffic', 'maximize': True},
    {'name': 'price', 'maximize': False},  # 价格越低越好
    {'name': 'region', 'maximize': True}
]

# 计算帕累托前沿
pareto_front = self.pareto_optimization(candidates, objectives)

# 从帕累托前沿中选择Top-K
# 方法1：按综合分数排序
final_sites = sorted(pareto_front, key=lambda x: x.final_score)[:top_k]

# 方法2：多样性采样
final_sites = self.diversity_sampling(pareto_front, top_k)
```

### 预期效果

- ✅ 提供多样化推荐
- ✅ 用户可根据偏好选择
- ✅ 更符合实际决策场景

---

## Phase 5: 增强可解释性（2-3天）

### 优化方案

```python
def generate_explanation(self, site_id, weights, scores):
    """
    生成推荐解释
    
    Returns:
        {
            "overall_score": 8.5,
            "breakdown": {
                "text": {"score": 8.5, "weight": 0.3, "contribution": 2.55},
                "traffic": {"score": 9.0, "weight": 0.4, "contribution": 3.60},
                "price": {"score": 7.5, "weight": 0.2, "contribution": 1.50},
                "region": {"score": 8.0, "weight": 0.1, "contribution": 0.80}
            },
            "strengths": ["交通便利（9.0分）", "地区优越（8.0分）"],
            "weaknesses": ["价格偏高（7.5分）"],
            "comparison": "比平均水平高15%"
        }
    """
    row = self.site_data.loc[site_id]
    
    # 计算贡献度
    breakdown = {}
    for dim, score in scores.items():
        weight = weights[dim]
        contribution = score * weight
        breakdown[dim] = {
            "score": round(score, 2),
            "weight": round(weight, 2),
            "contribution": round(contribution, 2)
        }
    
    # 识别优势和劣势
    strengths = [f"{dim}（{s['score']:.1f}分）" 
                 for dim, s in breakdown.items() if s['score'] >= 8.0]
    weaknesses = [f"{dim}（{s['score']:.1f}分）" 
                  for dim, s in breakdown.items() if s['score'] < 7.0]
    
    # 对比分析
    avg_score = self.site_data['final_score'].mean()
    diff_pct = ((scores['overall'] - avg_score) / avg_score) * 100
    comparison = f"比平均水平{'高' if diff_pct > 0 else '低'}{abs(diff_pct):.0f}%"
    
    return {
        "overall_score": round(sum(s['contribution'] for s in breakdown.values()), 2),
        "breakdown": breakdown,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "comparison": comparison
    }
```

### 前端展示

```javascript
// 分数拆解可视化
<div class="score-breakdown">
  <h4>综合评分: 8.5</h4>
  <div class="bar-chart">
    <div class="bar" style="width: 85%">
      <span>语义匹配 (8.5分, 权重30%)</span>
      <span class="contribution">贡献: 2.55</span>
    </div>
    <div class="bar" style="width: 90%">
      <span>交通便利 (9.0分, 权重40%)</span>
      <span class="contribution">贡献: 3.60</span>
    </div>
    <!-- ... -->
  </div>
  <div class="strengths">
    <strong>优势:</strong> 交通便利（9.0分）、地区优越（8.0分）
  </div>
  <div class="weaknesses">
    <strong>劣势:</strong> 价格偏高（7.5分）
  </div>
</div>
```

---

## Phase 6: 添加交互式调整（5-7天）

### 前端界面

```html
<!-- 权重调整滑块 -->
<div class="weight-controls">
  <h4>调整评分权重</h4>
  <div class="slider-group">
    <label>
      语义匹配 (<span id="w_text_val">30</span>%)
      <input type="range" id="w_text" min="0" max="100" value="30">
    </label>
    <label>
      交通便利 (<span id="w_traffic_val">40</span>%)
      <input type="range" id="w_traffic" min="0" max="100" value="40">
    </label>
    <label>
      性价比 (<span id="w_price_val">20</span>%)
      <input type="range" id="w_price" min="0" max="100" value="20">
    </label>
    <label>
      地区位置 (<span id="w_region_val">10</span>%)
      <input type="range" id="w_region" min="0" max="100" value="10">
    </label>
  </div>
  <button id="rerank">重新排序</button>
</div>
```

### 后端API

```python
@app.route('/api/rerank', methods=['POST'])
def rerank():
    """
    根据用户调整的权重重新排序
    """
    data = request.get_json()
    
    # 获取用户权重
    weights = {
        'text': data.get('w_text', 0.3),
        'traffic': data.get('w_traffic', 0.4),
        'price': data.get('w_price', 0.2),
        'region': data.get('w_region', 0.1)
    }
    
    # 归一化
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    # 重新计算分数
    candidates = data.get('candidates', [])
    for c in candidates:
        c['final_score'] = sum(
            weights[k] * c['scores'][k] 
            for k in weights
        )
    
    # 重新排序
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    return jsonify({
        'candidates': candidates,
        'weights': weights
    })
```

---

## 实施时间表

| Phase | 任务 | 工作量 | 优先级 | 预期完成 |
|-------|------|--------|--------|---------|
| **Phase 1** | 清理冗余 | 1-2天 | 🔥 高 | 第1周 |
| **Phase 2** | 简化评分 | 2-3天 | 🔥 高 | 第1周 |
| **Phase 3** | 增强权重推导 | 3-4天 | 🟠 中 | 第2周 |
| **Phase 4** | 多目标优化 | 4-5天 | 🟠 中 | 第2-3周 |
| **Phase 5** | 增强可解释性 | 2-3天 | 🟢 低 | 第3周 |
| **Phase 6** | 交互式调整 | 5-7天 | 🟢 低 | 第4周 |

**总计**：约3-4周

---

## 成功指标

### 代码质量
- [ ] 代码行数减少30%
- [ ] 圈复杂度降低50%
- [ ] 测试覆盖率达到80%

### 性能指标
- [ ] 响应时间 < 3秒
- [ ] 内存占用 < 500MB
- [ ] 并发支持 > 10 QPS

### 推荐质量
- [ ] 用户满意度 > 85%
- [ ] 推荐准确率 > 80%
- [ ] 多样性指标 > 0.7

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| LLM权重推导不稳定 | 中 | 中 | 添加降级策略，使用默认权重 |
| 多目标优化性能差 | 低 | 中 | 限制候选数量，使用启发式算法 |
| 前端交互复杂 | 中 | 低 | 提供预设模板，简化操作 |
| 用户不理解权重 | 高 | 低 | 添加帮助提示，提供示例 |

---

**下一步**：开始 Phase 1 - 清理冗余代码
