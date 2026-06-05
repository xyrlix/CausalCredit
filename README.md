# CausalCredit — 因果推理增强信用评分系统

> **不只告诉你风险有多高，更告诉你为什么高、以及如何降低**  
> 从相关性预测到因果性决策的范式跃迁

## 项目简介

CausalCredit 是一套面向金融机构的**因果推理增强信用评分系统**，参加**中银香港创新先驱大赛2026**。系统在传统机器学习预测基座之上，融合因果发现、异质处理效应估计（CATE）与反事实推理三大能力，构建"预测→归因→决策"的完整闭环。

## 核心技术亮点

| # | 亮点 | 技术方案 |
|---|------|---------|
| 1 | 混合因果发现引擎 | PC + NOTEARS 融合 + 领域知识注入 |
| 2 | CATE 异质处理效应估计 | EconML DML/DR/Causal Forest 三方法交叉验证 |
| 3 | 因果约束反事实推理 | DiCE + 因果图约束 + NSGA-II 多目标优化 |
| 4 | SHAP + 因果图联合可解释性 | 双层解释 + 四象限一致性校验 |
| 5 | 因果引导特征工程 | 因果路径强度 + 去混淆残差 + 中介效应分解 |

## 技术栈

- **因果推理**: DoWhy, EconML
- **预测模型**: LightGBM
- **可解释性**: SHAP, DiCE
- **后端**: FastAPI
- **前端**: Streamlit
- **部署**: Docker + Docker Compose

## 快速开始

### 环境要求

- Python 3.11+
- Docker（可选，用于容器化部署）

### 本地开发

```bash
# 安装依赖
make install

# 启动 API 服务
make run-api

# 启动 Streamlit Demo
make run-demo

# 运行测试
make test
```

### Docker 部署

```bash
docker-compose up --build
```

- API: http://localhost:8000/docs
- Demo: http://localhost:8501

## 项目结构

```
CausalCredit/
├── src/                    # 核心源代码
│   ├── data/               # 数据加载与校验
│   ├── features/           # 特征工程
│   ├── causal/             # 因果推理（图构建/效应估计/CATE）
│   ├── models/             # 预测模型（训练/评估/校准）
│   ├── explain/            # 可解释性（SHAP/反事实/决策建议）
│   ├── api/                # FastAPI 后端
│   ├── frontend/           # Streamlit 前端
│   └── utils/              # 工具模块
├── tests/                  # 测试
├── notebooks/              # Jupyter 探索笔记
├── configs/                # 配置文件
├── scripts/                # 工具脚本
└── docs/                   # 项目文档
```

## 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| Home Credit Default Risk | 30万+ 条 × 8表 | 主数据集 |
| Lending Club Loan Data | 200万+ 条 | 辅助验证 |
| German Credit Risk | 1,000 条 | 基线对比 |

## 许可证

MIT License
