# Material Desk demo

这是“企业技术资料与客户需求智能助手”的公开演示与后端部署子集。

在线访问：https://yghjnj.github.io/material-desk-demo/

[一键导入 Render 部署公网后端](https://render.com/deploy?repo=https://github.com/yghjnj/material-desk-demo)

- 仅使用项目内虚构演示资料
- 不连接企业内部资料、CRM 或外部客服平台
- 页面操作不会发送消息、报价、承诺交期或写入后台
- GitHub Pages 地址是静态演示，不接收或保存用户文件
- `demo/public_server.py` 提供会话隔离的公开后端：每个访客只能看到自己的资料，文件默认 24 小时后清理，支持 PDF/DOCX/MD/TXT 和 25 MiB 限制
- 用 Render 导入根目录 `render.yaml` 可部署网页和 Python API；平台会提供 HTTPS 地址。免费实例使用临时磁盘，适合演示，不适合作为企业长期知识库
- 本地开发仍可在项目目录内启动 `demo/server.py` 后使用 `http://127.0.0.1:8765/`
