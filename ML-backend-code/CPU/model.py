# -*- coding: utf-8 -*-
"""YOLO 自动标注后端

流程：/predict → 自动获取并解析 label studio XML 配置
     → 构造 dataclass 任务投递进程级流水线（接收→推理→后处理→返回）
     → 组装 ModelResponse 返回 label studio 前端
"""

import logging
from typing import Dict, List, Optional

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from Function import (
    后处理配置_默认,
    后处理器,
    批次状态,
    控件元数据,
    模型类别名,
    流水线任务,
    获取流水线,
)

logger = logging.getLogger(__name__)

# 共享后处理器实例：仅用于校验配置里的函数名是否存在（真正调用在流水线内部）
_后处理器 = 后处理器()


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        控件列表 = self._解析控件配置()
        if not tasks or not 控件列表:
            logger.warning("没有可处理的 task，或 XML 配置里没有已配置后处理函数的图像控件")
            return ModelResponse(predictions=[])

        logger.info(f"项目 {self.project_id}：{len(tasks)} 个 task、{len(控件列表)} 个控件进入推理流水线")
        批次 = 批次状态(
            待完成=len(tasks) * len(控件列表),
            结果=[[] for _ in tasks],
        )
        预测 = 获取流水线().提交(流水线任务(任务列表=tasks, 控件列表=控件列表, 批次=批次))

        # 返回给 label studio 前端
        for 单条预测 in 预测:
            单条预测["model_version"] = str(self.model_version)
        return ModelResponse(predictions=预测)

    def _解析控件配置(self) -> List[控件元数据]:
        """自动获取 label studio XML 配置

        每个图像控件：控件类型 → 配置里的函数名(str) → getattr 校验；
        同时用 XML 里的标签建立 模型类别 → 标注名 的映射。
        """
        界面 = getattr(self, "label_interface", None)
        if 界面 is None:
            return []
        控件列表: List[控件元数据] = []
        类别名列表: Optional[List[str]] = None
        for 控件 in 界面.controls:
            对象列表 = getattr(控件, "objects", None) or []
            if not 对象列表 or 对象列表[0].tag != "Image":
                continue  # 只处理图像控件

            函数名 = 后处理配置_默认.函数映射.get(控件.tag)
            if not 函数名:
                logger.warning(f"控件 {控件.tag}（name={控件.name}）没有配置后处理函数，已跳过；"
                               f"可在 后处理配置.函数映射 里添加")
                continue
            if not _后处理器.存在(函数名):
                logger.warning(f"后处理器里不存在函数 {函数名}（控件 {控件.tag}），已跳过")
                continue

            if 类别名列表 is None:
                类别名列表 = 模型类别名()
            控件列表.append(控件元数据(
                控件类型=控件.tag,
                控件名=控件.name,
                目标名=控件.to_name[0],
                图片字段=对象列表[0].value_name,
                标签映射=self.build_label_map(控件.name, 类别名列表),
                后处理函数名=函数名,
            ))
        return 控件列表

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        logger.info(f"收到事件 {event}：本后端为纯自动标注，不做增量训练")
