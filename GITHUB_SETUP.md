# AINEURO GitHub Repository / AINEURO GitHub仓库

This is the GitHub repository version of the AI Neuroscience (AINEURO) project.

这是AI神经科学（AINEURO）项目的GitHub仓库版本。

**Repository URL / 仓库地址**: https://github.com/linxiaocas/AINEURO-github/

**Contact / 联系方式**: Xiao.lin@ia.ac.cn

---

## Repository Structure / 仓库结构

```
AINEURO-github/
├── README.md                      # Project homepage / 项目主页
├── LICENSE                        # MIT License / MIT许可证
├── CONTRIBUTING.md                # Contributing guide / 贡献指南
├── CODE_OF_CONDUCT.md            # Code of conduct / 行为准则
├── CONTRIBUTORS.md               # Contributors list / 贡献者名单
├── .gitignore                    # Git ignore rules / Git忽略规则
├── 100篇论文目录.md               # Complete paper catalog / 完整论文规划
├── docs/                         # Documents / 文档
│   ├── 学科框架.md
│   ├── 完整学科框架_详细版.md
│   └── templates/
│       └── paper_template.md     # Paper template / 论文模板
├── papers/                       # Example papers / 示例论文
│   ├── 论文1_皮层微回路启发的深度学习架构.md
│   ├── 论文11_Transformer注意力与大脑前额叶工作记忆.md
│   ├── 论文21_突触缩放与神经稳态.md
│   ├── 论文31_高密度神经接口的实时解码.md
│   ├── 论文41_神经网络癫痫_过拟合与异常同步.md
│   ├── 论文51_神经进化的发育生物学启发.md
│   └── 论文71_AI意识的科学标准.md
├── theses/                       # Paper directions / 论文目录
├── src/                          # Tools / 工具代码
├── conferences/                  # Conference materials / 会议资料
├── talks/                        # Talk materials / 演讲资料
├── resources/                    # Resources / 资源
└── .github/                      # GitHub config / GitHub配置
    ├── ISSUE_TEMPLATE/
    ├── workflows/
    └── PULL_REQUEST_TEMPLATE.md
```

---

## Upload to GitHub / 上传到GitHub

### 1. Create GitHub Repository / 创建GitHub仓库

1. Login to GitHub / 登录GitHub
2. Click "New repository" / 点击 "New repository"
3. Repository name / 仓库名：`AINEURO-github`
4. Select "Public" or "Private" / 选择 "Public" 或 "Private"
5. **Do not** initialize README / **不要**初始化README（我们已经有了）
6. Click "Create repository" / 点击 "Create repository"

### 2. Push to GitHub / 推送到GitHub

```bash
# Enter repository directory / 进入仓库目录
cd AINEURO-github

# Initialize git repository / 初始化git仓库
git init

# Add all files / 添加所有文件
git add .

# Commit / 提交
git commit -m "Initial commit: AINEURO disciplinary framework / 初始提交：AINEURO学科体系"

# Link remote repository / 关联远程仓库
git remote add origin https://github.com/linxiaocas/AINEURO-github.git

# Push / 推送
git push -u origin main
```

### 3. Set up GitHub Pages (Optional) / 设置GitHub Pages（可选）

To showcase documents using GitHub Pages / 如需使用GitHub Pages展示文档：

1. Go to Settings > Pages / 进入仓库 Settings > Pages
2. Source: Select "Deploy from a branch" / Source 选择 "Deploy from a branch"
3. Branch: Select "main", folder: "/docs" / Branch 选择 "main"，文件夹选择 "/docs"
4. Click Save / 点击 Save

---

## Subsequent Development / 后续开发

### Add New Paper / 添加新论文

```bash
# Create new branch / 创建新分支
git checkout -b paper/direction_02_attention_mechanism

# Write paper / 撰写论文
echo "# New Paper Title / 新论文标题" > theses/direction_02_cognition/论文15_注意力机制的新理论.md

# Commit / 提交
git add .
git commit -m "paper(cognition): add attention mechanism theory"

# Push / 推送
git push origin paper/direction_02_attention_mechanism

# Create Pull Request / 创建Pull Request
```

### Update Documentation / 更新文档

```bash
git checkout -b docs/update_neuroarchitecture

# Edit documents / 修改文档
vim docs/完整学科框架_详细版.md

git add .
git commit -m "docs(arch): update neuroarchitecture chapter"
git push origin docs/update_neuroarchitecture
```

---

## Features / 功能特性

- ✅ Complete disciplinary framework / 完整的学科框架文档
- ✅ 100 paper planning / 100篇论文规划
- ✅ 7 detailed example papers (bilingual) / 7篇详细示例论文（双语）
- ✅ GitHub Actions automation / GitHub Actions自动化
- ✅ Issue/PR templates / Issue/PR模板
- ✅ Contributing guide and code of conduct / 贡献指南和行为准则
- ✅ MIT License / MIT许可证
- ✅ Bilingual documentation (English/Chinese) / 双语文档（中英文）

---

## Contact / 联系方式

- **GitHub**: https://github.com/linxiaocas/AINEURO-github/
- **Email**: Xiao.lin@ia.ac.cn
- **Maintainer**: Xiao Lin

---

**Let's explore the essence of intelligence together, connecting silicon and carbon life!**

**让我们共同探索智能的本质，连接硅基与碳基生命！**
