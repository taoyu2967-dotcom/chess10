# 安全审计说明（SECURITY.md）

> 归档前审计：2026-09-12，Mimosa 深度安全扫描（normal 深度）
> scanId: `scan-2026-09-12T07-36-23.585Z-8994ca917de6`
> seal: `sha256:f6c43464fc4e8018df67225bc85860564a3dbab1581196de9adbfa19d6a7cbca`
> 结果：57 条发现（39 high / 18 medium），依赖风险 0（无受影响依赖包）

## 结论

经项目所有者逐条复核后确认：**本仓库为本地单用户工具集与训练套件，全部发现均属"扫描器模式命中、上下文不成立"**，按现状归档。逐类分诊如下。

## 分诊明细

### 1. "命令注入"（execSync/exec 模板串，~15 处 high）

代表：`training/boot_daemon.js`、`server/selfplay_fsf_teacher.js:120`、`server/tools/*`。

被拼进命令串的实参全部是**硬编码常量路径**（如 `path.join(T,'start_daemon.ps1')`，T 为写死的绝对路径）或**进程内数字**（如 `p.pid`、`1 << coreId`）。不存在任何外部输入进入命令串的通路。修复方向（改 execFileSync 参数数组）对安全性无实质增益，属机械改写。

### 2. "不可信程序选择"（3 处 high）

`server/selfplay_fsf_teacher.js:44`、`server/server.js`、`server/chess10_server.js`：spawn 的可执行文件路径支持 `CHESS10_FSF` 等环境变量覆盖——这是**有意的可移植性设计**（云端 Linux 与本地 Windows 共用同一套代码），环境变量由本机使用者控制，不构成攻击面。

### 3. "代码注入"（vm.runInContext / eval，6 处 high）

`server/tools/balance_trace.js`、`ghost_pawn_trace.js`、`html_engine_*.js`、`repro_ai_vs_ai.js`：这些是**开发者自用的公式/策略求值沙箱**，"被注入的代码"即开发者本人在命令行输入的表达式，沙箱限制全局对象正是为了隔离副作用。无远程/不可信输入源。

### 4. "路径穿越"（7 处 high）

`ov_train/export_*.py`、`loss_surface*.py`、`server/ov_bridge_server_v2.py`、`ov_nncf_int8.py`：这些 CLI 工具的设计功能就是**接收使用者指定的任意路径**（权重/数据文件可以在任意盘符），"穿越"即正常功能。

### 5. "疑似跨文件污点"（14 处 medium）与 `lib/stockfish.js` 的"MongoDB 注入"（medium）

跨文件污点为启发式告警，本仓库无外部输入源；`lib/stockfish.js` 是第三方国际象棋引擎的标准发行文件，其中的"MongoDB 排序字段"命中为字符串模式巧合（该文件根本不使用 MongoDB）。

## 风险边界（诚实声明）

- 唯一长期监听网络的进程是 `server.js`（本地对弈服务）。着法等用户输入经由**stdio 管道**传给引擎子进程，不经过 shell 拼接；但项目整体从未按"暴露公网的多用户服务"做过加固，**请勿将 server.js 直接暴露到不可信网络**。
- `fsf/fsf-server.js` 在 Mimosa 报告中命中 5 处 high，该文件**不在本仓库内**（fsf/ 目录仅提交 variants.ini 变体定义）。
- 仓库为私有仓库。若将来转为公开或部署到服务器，应重新评估第 1、3 类发现。
