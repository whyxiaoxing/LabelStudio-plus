# Label Studio

数据标注工具。后端 Django + DRF，前端 React，许可证 Apache-2.0。

当前版本 `1.24.0.dev0`。本代码树是**社区版（开源版）**；企业版是另一套闭源发行物，不包含在这里。

---

## 从源码安装

Python 包本身可以直接安装，但**必须先构建前端** —— Django 的标注编辑器和 Data Manager
是直接从 `web/dist/` 读取的，而刚克隆下来的代码树里没有 `web/dist/`：

```bash
# 1. 构建前端（需要 bun）
cd web
bun install
bun run build          # 产出 web/dist/{apps/labelstudio,libs/editor,libs/datamanager}

# 2. 安装后端
cd ..
pip install .
label-studio
```

开发时改用 `cd web && bun run dev`：它会启动带 HMR 的 Vite 开发服务器并反向代理到
Django（由 `DJANGO_HOSTNAME` 指定，默认 `http://localhost:8080`）。这种方式下 Django
不需要 `web/dist`。

### 分发形态：请用源码包 / sdist，不要用 wheel

`settings.EDITOR_ROOT`、`DM_ROOT`、`REACT_APP_ROOT`（见
`label_studio/core/settings/base.py`）都是**按仓库目录结构**推导出来的，指向
`<仓库根>/web/dist/...`；`core/utils/manifest_assets.py` 也会去读
`STATIC_ROOT/js/manifest.json`。

只要保持仓库目录结构，这些路径就能正确解析 —— **sdist 会保留目录结构，所以没问题**。
而 wheel 会把文件装到 `site-packages/` 下，上述路径全部失效，界面加载不出来。

---

## 目录结构

| 路径 | 内容 |
| :--- | :--- |
| `label_studio/` | Django 后端（`core`、`projects`、`tasks`、`data_export`、`io_storages`、`ml` 等） |
| `web/` | 前端 monorepo —— `apps/labelstudio`（主应用）、`libs/editor`（标注界面）、`libs/datamanager`、`libs/ui`、`libs/core` |
| `label_studio_sdk/` | 内置的 `label-studio-sdk`（含本地补丁），随本发行包一起打包，详见下节 |
| `docs/source/` | 文档源码 |
| `deploy/` | 辅助脚本 |

### 内置的 SDK

`label_studio_sdk/` 是 `label-studio-sdk` 2.1.2 的完整副本，放在仓库内是为了让本地修复
在重装依赖后依然存在。

它**没有**被声明为依赖项 —— 如果再另外从 PyPI 装一份 `label-studio-sdk`，会因为导入顺序
把这份带补丁的副本遮蔽掉。它通过 `[tool.setuptools.packages.find]` 和 `MANIFEST.in`
随本发行包一起发布。

改它的时候注意：`converter/converter.py` 会从
`_extensions/label_studio_tools/core/utils/io.py` 导入文件名 / URI 处理函数，
**这两个文件必须一起改**。

---

## 测试

```bash
pip install --group test      # 或者：pip install -r <你自己的锁定清单>
DJANGO_DB=sqlite python -m pytest label_studio -q
```

前端有独立的测试套件：`cd web && bun run test:unit`。

---

## 许可证

Apache-2.0。
