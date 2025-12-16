# Repository Guidelines

## Project Structure & Module Organization
- `capture_image_worker.py` 是实时采集和缺陷检测的主工人进程，协调 `camera`, `config`, `utils` 等同级模块并写入 `data/` 与 `Path.IMAGE_SAVE_DIR`。调整硬件参数前务必更新 `Path.CCF_DIR` 下的 ccf 模板以避免设备失配。
- `detect_core/` 存放可复用的算法库：`castfilm_detector.py` 与 `statistics_calculator.py` 提供高阶接口；`castfilm_detection/` 内部的 `common/` 负责文件系统工具，`processing/` 下的 `background.py`, `filters.py`, `membrane.py`, `defects.py` 描述各个流水线 stage。
- `data/` 仅用于示例帧或离线回放，体积较大。请将新的训练或验证样本置于子目录并附上 `README` 说明采集条件；正式发布前不要提交含客户信息的原图。

## Build, Test, and Development Commands
- 使用虚拟环境保证依赖隔离：`python -m venv .venv`，随后 `.venv\Scripts\activate` 与 `pip install -U numpy opencv-python pillow loguru aiofiles`.
- 离线批处理检测：`python -m detect_core.castfilm_detection.pipeline --input data --output output --min-component 12 --blur-mode opencv --workers 4 --pool thread`，该命令会在 `output/` 写入标注图与 `detection_report.txt`。
- API 调试示例：`python -c "from detect_core.castfilm_detection import api; api.analyze_image_file('data/1.jpg', save_annotation=True, output_dir='output')"`，适合验证单帧阈值与包围盒。
- 采集进程需具备硬件：`python capture_image_worker.py`，启动前确认 `camera/` DLL 与 CCF 配置存在并在同级 PYTHONPATH 中。

## Coding Style & Naming Conventions
- Python 代码统一使用 4 空格缩进、`snake_case` 函数、`PascalCase` 类名、全局常量大写（参考 `Constants.MAX_BUFFER_IMAGE`）。建议所有新模块补充 `from __future__ import annotations` 与类型注解。
- 未提供强制格式化配置时，请运行 `black . --line-length 100` 与 `ruff check .`（或等效工具）后再提交，保证 import 顺序与 lint 整洁。

## Testing Guidelines
- 首选 `pytest`，在 `detect_core/tests/`（如不存在可新建）按 `test_<module>.py` 命名，并使用 NumPy 生成的合成灰度阵列或 `data/` 缩放片段，避免依赖整张 30MB 图。
- 新增处理步骤需补充边界案例（如膜宽不足、`blur_mode` 回退等），并在 MR/PR 中附上 `pytest -q detect_core --maxfail=1` 的输出或截图；涉及统计指标的更改应包含 ≥80% 语句覆盖率的测试。

## Commit & Pull Request Guidelines
- 本仓库尚未初始化 Git 历史，建议遵循 Conventional Commits（例如 `feat: add integral blur fallback`、`fix: clamp membrane bounds`）并在正文描述动机、关键变更与验证手段。
- PR 描述至少包含：需求或关联问题链接、影响面、手动或自动测试记录、如有 UI/可视化变动须附前后对比图；确保 reviewers 能在 `Steps to Reproduce` 中复现你的结果。

## Security & Configuration Tips
- `capture_image_worker.py` 会访问生产相机与灯控，提交代码前请清理日志中的序列号或客户参数；所有硬件密钥放入未跟踪的 `.env` 或 Windows Credential Manager。
- 任何修改 `Path.CCF_DIR`, `Path.IMAGE_SAVE_DIR`, `Path.CAMERA_DIR` 的改动需在说明文档中注明，以便现场人员同步配置；上传敏感数据前先执行 `git lfs track "*.jpg"` 或改为脱敏缩略图。
