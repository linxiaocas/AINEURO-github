# AINEURO GitHub 仓库

这是AI神经科学（AINEURO）项目的GitHub仓库版本。

## 仓库结构

```
AINEURO-github/
├── README.md                      # 项目主页
├── LICENSE                        # MIT许可证
├── CONTRIBUTING.md                # 贡献指南
├── CODE_OF_CONDUCT.md            # 行为准则
├── CONTRIBUTORS.md               # 贡献者名单
├── .gitignore                    # Git忽略规则
├── 100篇论文目录.md               # 完整论文规划
├── docs/                         # 文档
│   ├── 学科框架.md
│   ├── 完整学科框架_详细版.md
│   └── templates/
│       └── paper_template.md     # 论文模板
├── papers/                       # 示例论文
│   ├── 论文1_皮层微回路启发的深度学习架构.md
│   ├── 论文11_Transformer注意力与大脑前额叶工作记忆.md
│   ├── 论文41_神经网络癫痫_过拟合与异常同步.md
│   └── 论文71_AI意识的科学标准.md
├── theses/                       # 论文目录（10个方向）
│   ├── direction_01_architecture/
│   ├── direction_02_cognition/
│   ├── direction_03_plasticity/
│   ├── direction_04_bci/
│   ├── direction_05_pathology/
│   ├── direction_06_evolution/
│   ├── direction_07_multimodal/
│   ├── direction_08_ethics/
│   ├── direction_09_signal/
│   └── direction_10_quantum/
├── src/                          # 工具代码
│   └── README.md
├── conferences/                  # 会议资料
├── talks/                        # 演讲资料
├── resources/                    # 资源
└── .github/                      # GitHub配置
    ├── ISSUE_TEMPLATE/           # Issue模板
    ├── workflows/                # CI/CD工作流
    └── PULL_REQUEST_TEMPLATE.md  # PR模板
```

## 上传到GitHub

### 1. 创建GitHub仓库

1. 登录GitHub
2. 点击 "New repository"
3. 仓库名：aineuro
4. 选择 "Public" 或 "Private"
5. 不要初始化README（我们已经有了）
6. 点击 "Create repository"

### 2. 推送到GitHub

```bash
# 进入仓库目录
cd AINEURO-github

# 初始化git仓库
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: AINEURO学科体系"

# 关联远程仓库（替换 YOUR_USERNAME 为您的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/aineuro.git

# 推送
git push -u origin main
```

### 3. 设置GitHub Pages（可选）

如需使用GitHub Pages展示文档：

1. 进入仓库 Settings > Pages
2. Source 选择 "Deploy from a branch"
3. Branch 选择 "main"，文件夹选择 "/docs"
4. 点击 Save

## 后续开发

### 添加新论文

```bash
# 创建新分支
git checkout -b paper/direction_02_attention_mechanism

# 撰写论文
echo "# 新论文标题" > theses/direction_02_cognition/论文15_注意力机制的新理论.md

# 提交
git add .
git commit -m "paper(cognition): add attention mechanism theory"

# 推送
git push origin paper/direction_02_attention_mechanism

# 创建Pull Request
```

### 更新文档

```bash
git checkout -b docs/update_neuroarchitecture

# 修改文档
vim docs/完整学科框架_详细版.md

git add .
git commit -m "docs(arch): update neuroarchitecture chapter"
git push origin docs/update_neuroarchitecture
```

## 功能特性

- ✅ 完整的学科框架文档
- ✅ 100篇论文规划
- ✅ 4篇详细示例论文
- ✅ GitHub Actions自动化
- ✅ Issue/PR模板
- ✅ 贡献指南和行为准则
- ✅ MIT许可证

## 联系方式

- GitHub: https://github.com/aineuro/aineuro
- Email: contact@aineuro.org

---

**让我们共同探索智能的本质，连接硅基与碳基生命！**
