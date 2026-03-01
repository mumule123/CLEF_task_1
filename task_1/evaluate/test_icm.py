from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
import os

# 使用os.path处理路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 文件路径
gold = os.path.join(project_root, "datasets", "EXIST2025_dev_gold_all.json")
predictions = os.path.join(project_root, "model_out", "task1_hard_normalized_6.json")

# 初始化评估器
evaluator = PyEvALLEvaluation()

# 设置参数
# params = {
#     PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED
# }
params= dict()
params[PyEvALLUtils.PARAM_REPORT]= PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME


# 指定使用的指标
metrics = ["ICM", "ICMNorm", "FMeasure"]

# 执行评估
report = evaluator.evaluate(predictions, gold, metrics, **params)

# 输出评估结果
report.print_report()
