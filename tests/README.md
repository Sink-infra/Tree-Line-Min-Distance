# TreeLineMinDistance 测试

本目录按 Coverage、Correctness、Mock 三类组织测试。案例参数保存在 `st/cases/` 的三份 JSON 中；`common/` 提供 Case 类型、分类加载、确定性输入、数学基准和张量断言，`ut/` 检查这些模块，`st/scripts/` 提供数据生成、当前构建的 NPU 执行和性能采样入口。

| 类别 | 案例文件 | 数量 | 用途 |
|---|---|---:|---|
| Coverage | `st/cases/coverage.json` | 48 | 20 个合法输入和 28 个输入契约拒绝边界 |
| Correctness | `st/cases/correctness.json` | 35 | 有限线段几何、平局、尾块及精度 |
| Mock | `st/cases/mock.json` | 17 | 固定工作负载形状和类型 |

`full` 为 Coverage 与 Correctness 的 83 项并集，`all` 包含三类共 100 项。
全部案例中有 72 个合法输入和 28 个拒绝边界。

## 目录与职责

```text
tests/
├── README.md
├── __init__.py
├── __main__.py
├── requirements.txt
├── common/
│   ├── cases.py                 # CaseSpec、JSON 加载和分类选择
│   ├── reference.py             # 确定性输入、契约校验和 FP64 数学基准
│   ├── tensor_io.py             # 二进制存储、字节数和文件摘要
│   ├── generate_data.py         # Fixture、数学基准文件和元数据
│   ├── verify_result.py         # 距离与索引的逐元素断言
│   ├── checks.py                # 声明和生成数据的共用自检
│   ├── device.py                # 当前构建执行、文件摘要和设备结果
│   └── profiler.py              # 任务采样和多进程统计
├── st/
│   ├── cases/
│   │   ├── coverage.json
│   │   ├── correctness.json
│   │   └── mock.json
│   ├── runner/
│   │   └── tree_line_min_distance_runner.cpp
│   └── scripts/
│       ├── gen_data.py          # Case 列表和数据生成
│       ├── build.py             # 当前算子构建与产物摘要
│       ├── run_tests.py         # NPU 输出断言
│       └── profile_mock_workloads.py # Mock 任务计时
└── ut/
    ├── test_case_loading.py
    ├── test_case_matrix.py
    ├── test_reference.py
    ├── test_mock_cases.py
    ├── test_execution.py
    └── test_batch_execution.py
```

| UT 文件 | 内容 |
|---|---|
| `test_case_loading.py` | JSON 配置、分类隔离、名称及标签选择、ST 生成入口 |
| `test_case_matrix.py` | Coverage 声明、输入范围、拒绝边界和数据断言 |
| `test_reference.py` | 解析几何、独立标量公式、归约、量化和输出断言 |
| `test_mock_cases.py` | Mock 声明及全部工作负载的数据和基准 |
| `test_execution.py` | 执行结果、构建摘要、采样完整性和多进程统计 |
| `test_batch_execution.py` | 批次停止、冷启动结果、混合 dtype 采样归属和完成记录 |

## Case 配置

测试契约由 [TreeLineMinDistance.md](../TreeLineMinDistance.md) 定义。
三份配置都使用 `schema_version=1` 和 `operator=TreeLineMinDistance`，根字段 `category` 对应文件名。

每个 Case 声明以下内容：

| 字段 | 含义 |
|---|---|
| `name` | 分类前缀和唯一名称 |
| `expectation` | `success` 或 `reject` |
| `tree_points_shape`、`line_segments_shape` | 两个输入的形状 |
| `dtype`、`line_segments_dtype` | 输入类型；线段类型缺省时与点相同 |
| `input_pattern`、`seed` | 几何构造与固定种子 |
| `tree_points_format`、`line_segments_format` | 输入格式 |
| `expected_error` | 拒绝 Case 的输入契约标签 |
| `covers`、`note` | 覆盖标签和用例说明 |

加载器检查分类、字段、名称、整数形状和种子编码。拒绝 Case 的非法算子输入规格保留在配置中，由输入契约断言检查。

Coverage 对每种 dtype 覆盖：

```text
(1, 1, 1)          (1, 3, 7)          (2, 19, 19)
(3, 33, 31)        (4, 127, 33)       (2, 1003, 129)
(1, 4097, 513)     (1, 16384, 1)      (8, 1, 1024)
(8, 16384, 1024)
```

28 个拒绝边界涉及 Rank、坐标维、端点维、批次不一致、零元素、B/N/M 超限、输入类型不一致、不支持的类型或格式，以及两个输入各自的 NaN/+Inf/-Inf。
坐标 `[-1000,1000]` 是建议范围，输入契约不将它作为硬性边界。

Correctness 包含题面示例、斜向投影、端点、负坐标、零长度线段、最小索引平局、跨块尾部、批次隔离及 FP16 中间精度。
FP16 专项检查大坐标、微小非零线段、输出量化平局和端点量化退化。

## Case 选择与数据生成

在仓库根目录执行：

```bash
python3 -m tests --details
python3 -m tests.st.scripts.gen_data --suite coverage --list
python3 -m tests.st.scripts.gen_data --suite correctness --tag zero_length --list
python3 -m tests.st.scripts.gen_data --suite mock --list
```

`--suite` 支持 `coverage`、`correctness`、`mock`、`full`、`all` 和 `custom`。
`--case` 按完整名称选择，可重复指定；命名 Case 必须属于选定的套件。
`custom` 必须提供至少一个 Case 名称，可跨分类选择。
`--tag` 要求案例同时包含所有给定的 `covers` 标签。
`--expectation` 支持 `all`、`success` 和 `reject`。

```bash
python3 -m tests.st.scripts.gen_data --suite coverage --output-dir build/tests/data/coverage
python3 -m tests.st.scripts.gen_data --suite correctness --output-dir build/tests/data/correctness
python3 -m tests.st.scripts.gen_data --suite mock --output-dir build/tests/data/mock
python3 -m tests.st.scripts.gen_data --suite full \
  --case coverage_float16_singleton \
  --case correctness_float32_documented_endpoints \
  --output-dir build/tests/data/selected
python3 -m tests.st.scripts.gen_data --suite coverage --expectation reject \
  --output-dir build/tests/data/rejected
```

`--output-dir` 指定尚未存在的目录，`--output` 使用相同参数。
`--list` 只显示案例，不生成文件。生成的数据放在被 Git 忽略的 `build/` 中。

## 数学基准与输出断言

输入按声明类型量化为 FP16/FP32，再使用实际坐标计算 FP64 基准。
参考公式投影到有限线段，参数限制在 `[0,1]`；零长度线段跳过除法。
点和线段分别按最多 256、128 项分块，维护最小距离平方，最终对胜出距离开平方。
等距时取最小线段索引。

| 距离 dtype | atol | rtol | 索引断言 |
|---|---:|---:|---|
| float32 | 1e-4 | 1e-4 | 完全一致 |
| float16 | 1e-3 | 1e-3 | 完全一致 |

每个距离必须有限、非负，并满足 `abs(actual-reference) <= atol+rtol*abs(reference)`。
索引必须与基准一致且处于 `[0,M-1]`。平均误差不能代替逐元素断言。

输入和基准文件检查编码、字节数及 SHA256；结果文件检查精确字节数。
自检使用独立标量公式及不同分块交叉检查小形状。
对于分离线段构造的大形状，按已知最近线段独立复算每个点的距离和索引。

## Fixture 文件

每个合法 Case 包含：

```text
case.json
tree_points.bin
line_segments.bin
reference_min_distance_fp64.bin
reference_nearest_segment_idx_int32.bin
```

二进制文件为小端序、C 顺序连续存储。
点输入为 `[B,N,3]`，线段为 `[B,M,2,3]`，距离及索引基准均为 `[B,N]`。
拒绝 Case 只包含输入和元数据。

`case.json` 记录实际输入规格、生成参数、比较规则和文件摘要。
根目录 `manifest.json` 记录所选套件、Case 元数据摘要、数量及 Python/NumPy 版本。
组合套件的 manifest 采用组合名称，各 Case 的元数据保留自己的分类。

## 固定 Mock

| Case | B | N | M | 输入及距离 dtype |
|---|---:|---:|---:|---|
| mock_01 | 1 | 1 | 1 | float32 |
| mock_02 | 1 | 19 | 7 | float32 |
| mock_03 | 1 | 33 | 31 | float16 |
| mock_04 | 8 | 3 | 19 | float32 |
| mock_05 | 4 | 127 | 129 | float32 |
| mock_06 | 2 | 256 | 64 | float16 |
| mock_07 | 2 | 1003 | 513 | float32 |
| mock_08 | 2 | 4097 | 513 | float16 |
| mock_09 | 1 | 16384 | 1 | float32 |
| mock_10 | 1 | 1 | 1024 | float16 |
| mock_11 | 8 | 16384 | 1024 | float32 |
| mock_12 | 4 | 8192 | 1024 | float16 |
| mock_13 | 1 | 1 | 2 | float32 |
| mock_14 | 1 | 1 | 2 | float16 |
| mock_15 | 1 | 3 | 2 | float32 |
| mock_16 | 1 | 3 | 1 | float32 |
| mock_17 | 1 | 1 | 6 | float32 |

Mock 使用固定种子 `20261004+Case 编号` 和合成几何坐标。
斜向线段位于分离的 x 单元中，点在线段附近分布，最近索引随点号和批次变化。

## NPU 执行与性能采样

ST 在 Ascend Linux ARM64 目标上执行，使用 Ascend 910B 和 CANN 9.0.0。
构建、数据、实际输出、日志及性能报告位于被 Git 忽略的 `build/` 或仓库外的独立目录。
每次执行使用一个尚不存在的结果目录。

```bash
python3 -m tests.st.scripts.build --build-dir build/tests/current
python3 -m tests.st.scripts.run_tests \
  --build-dir build/tests/current --suite full \
  --output-dir build/tests/execution
```

`--source-dir` 默认为 `code/`，`--toolkit` 默认为 `/usr/local/Ascend/cann-9.0.0`。
构建摘要覆盖源码、算子库、完整内核包及生成头文件。执行前后核对构建产物、输入、基准及 Case 顺序；
源码变动后需要重新构建。Runner 直接链接 ACLNN 接口，并核对实际加载的算子库路径。

普通执行按所选 Case 顺序运行，共享一个 ACL 会话和设备流。
首个运行错误保留具体 API 返回码与阶段，后续合法案例记为 Skipped。
数值错误由 Python 逐元素校验，其他案例仍正常执行。
输入契约拒绝案例由 UT 检查，设备报告明确记为 Skipped；只有拒绝案例的选择不能作为设备执行通过。

每个设备张量的数据区前后各保留 512 字节保护区，记录分配器提供的实际对齐，地址至少按 32 字节对齐。
正确性阶段每次调用前设置输出哨兵，检查首次和末次输出、设备输入及保护区。
缺失或截断输出、异常进程退出及批次记录不完整均不能判为通过。
`results.json` 保存批次过程及各 Case 的状态、错误码、输出断言和逐元素错误比例。

```bash
python3 -m tests.st.scripts.profile_mock_workloads \
  --build-dir build/tests/current \
  --warmup 5 --repeat 5 --processes 3 \
  --output-dir build/tests/performance

python3 -m tests.st.scripts.profile_mock_workloads \
  --build-dir build/tests/current --case mock_01 --case mock_03 \
  --output-dir build/tests/performance-selected
```

性能入口默认运行全部 17 个 Mock；也支持套件、名称和标签选择合法输入案例。
`--device` 默认为 0，Profiler `--warmup` 和每例 `--repeat` 默认为 5，
`--processes` 默认为 1，`--timeout` 是每个完整设备批次的期限，默认为 300 秒。
每个计时进程包含完整所选序列，共享单一会话；首个 RE 后停止该批次。

每轮先进行独立的冷启动正确性批次，再进行 Profiler 计时批次。两者使用同一构建包和输入。
冷启动输出必须通过检查，计时批次最终输出也必须通过检查。
计时热循环不重置输出、不复制中间检查数据；每个 Case 的初始哨兵、最终输入和保护区仍需验证。
报告 `first` 来自冷启动批次，`final` 来自计时批次，两个阶段的完整结果分别保存在
`correctness`、`measurement` 字段及目录中。

`msprof op` 为每次采样执行指定次数的预热。
从 `OpBasicInfo` CSV 读取 `TaskDuration(us)`，按当前构建的内核配置识别 dtype，
按每个内核的数值启动序号及批次 Case 顺序分配采样。
缺失、重复、额外或无法明确归属的样本使性能采样失败。
每轮采样上限为所选 Case 数量乘以 `--repeat`，不得超过 5000。

`performance.json` 保留全部原始样本、CSV 摘要、进程内均值/中位数/最小值/最大值。
每个 Case 的 `timing.elapsed_us` 是多个进程样本均值的中位数；
只有一个进程时就是该 Case 全部采样的算术均值。
该时间不包含 Python 调度、数据生成、进程启动或输出校验。
数值错误、API 失败及无效采样均以非零状态退出；可获得的失败案例时间仅作为诊断数据。
主机上的套件执行器共享设备锁，正确性和计时批次之间也不会相互重叠。

## UT 与依赖

Python 3.9 或更高版本，依赖为 Python 标准库和 NumPy：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r tests/requirements.txt
.venv/bin/python -m unittest discover -s tests/ut -t . -v
```

UT 检查全部 100 个 Case，包括最大工作负载和输入拒绝边界，并覆盖数据篡改、距离或索引错误、文件截断及错误的配置。
