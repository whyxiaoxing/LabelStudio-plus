# -*- coding: utf-8 -*-
"""推理函数实现：YOLO 推理、后处理与进程级流水线

结构（对应 README TODO）：
1. 推理参数 / 后处理配置：dataclass，只放"开放"的参数，用户直接改字段调试；
2. 推理()：唯一真正调用 YOLO 的位置；不开放参数（device=cpu、imgsz=960）
   写死在内部的 model.predict 调用里，其余参数全部来自 推理参数；
   单张图进、单张图出，predict 完立刻用 转numpy() 转成纯 numpy 的单图结果
   （只用 numpy 接口转换，不使用 torch 专有接口）；
3. 后处理器：每个 xxx_后处理 函数对应 label studio 的一种控件类型，
   通过配置里的字符串（后处理函数映射）用 getattr 取出，可手动修改；
   输入永远是纯 numpy 的单图结果，输出是 label studio 固定结果格式；
4. 推理流水线：进程级常驻线程池（max_thread=6）：
   接收×1 → 推理×1 → 后处理×1-2 → 返回×1，线程间全部用 Queue 传递 dataclass；
   每个阶段先筛选有效数据再交给下一级：拿不到图片 / 推理失败 / 没有检测的单元
   就地按空结果登记后结束，不进入下一个线程。
"""

import atexit
import logging
import os
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.converter.brush import mask2rle
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# 线程池上限：接收 1 + 推理 1 + 后处理 1-2 + 返回 1 = 4-5 常驻，加模型预热 1 = 6
最大线程数 = 6
# 后处理线程数（README：1-2 个）
后处理线程数 = 2
# 单次 predict 等待流水线结果的最长秒数
流水线超时 = 600.0


# ---------------------------------------------------------------------------
# 一、配置（dataclass）
# ---------------------------------------------------------------------------
@dataclass
class 推理参数:
    """YOLO 推理的开放参数：需要调试时直接改这里的默认值

    不开放的参数只有 device=cpu、imgsz=960，它们写死在
    推理() 内部的 model.predict 调用里，不在这里出现。
    """

    模型路径: str = "yolo11n.pt"          # 换成自己的权重文件即可
    置信度: float = 0.25                  # ultralytics: conf
    IOU阈值: float = 0.7                  # ultralytics: iou
    最大检测数: int = 300                 # ultralytics: max_det
    类别索引: Optional[List[int]] = None  # ultralytics: classes，只保留部分类别时填
    类别无关NMS: bool = False             # ultralytics: agnostic_nms
    额外参数: Dict[str, Any] = field(default_factory=dict)  # 其余 ultralytics 参数从这里透传


@dataclass
class 后处理配置:
    """XML 控件类型 → 后处理器内函数名的映射（就是 README 里那个 str）

    运行时用 getattr(后处理器, 函数名) 取出函数使用；
    新增控件类型时：在 后处理器 里写一个 xxx_后处理 函数，再在下面补一行映射。
    """

    函数映射: Dict[str, str] = field(default_factory=lambda: {
        "RectangleLabels": "框_后处理",      # 矩形框
        "BrushLabels": "掩码_后处理",        # 掩码（RLE）
        "PolygonLabels": "多边形_后处理",    # 掩码（多边形点集）
    })


推理参数_默认 = 推理参数()
后处理配置_默认 = 后处理配置()


# ---------------------------------------------------------------------------
# 二、流水线消息（全部是 dataclass，在线程间通过 Queue 传递）
# ---------------------------------------------------------------------------
@dataclass
class 控件元数据:
    """从 label studio XML 配置里解析出的一个标注控件"""

    控件类型: str                    # 如 RectangleLabels / BrushLabels
    控件名: str                      # from_name
    目标名: str                      # to_name
    图片字段: str                    # task['data'] 里取图片的字段，如 image
    标签映射: Dict[str, str]         # 模型类别名 → label studio 标注名
    后处理函数名: str                # 配置映射出的字符串，getattr 取函数


@dataclass
class 单图结果:
    """单张图的推理结果：字段全是纯 numpy 数组 / 内置类型

    在推理线程里由 转numpy() 生成，之后线程间传递不含任何 torch 张量，
    后处理只需 numpy 即可完成。
    """

    类别下标: np.ndarray                        # (N,) 每个检测的类别索引
    置信度: np.ndarray                          # (N,) 每个检测的置信度
    归一化框: np.ndarray                        # (N, 4) xywhn：归一化中心点 + 宽高
    掩码: Optional[np.ndarray] = None           # (N, h, w) 检测掩码（仅分割模型有）
    多边形: Optional[List[np.ndarray]] = None   # 每条检测的归一化轮廓点集 (P, 2)
    原图高: int = 0
    原图宽: int = 0
    类别名: Dict[int, str] = field(default_factory=dict)

    @property
    def 检测数(self) -> int:
        """本图检测到的目标数（0 表示没有有效检测，流水线会就地按空结果登记）"""
        return int(self.类别下标.shape[0])


@dataclass
class 批次状态:
    """一次 /predict 请求的共享状态：任何阶段都可能登记完成，登记动作由锁保护

    每份（任务 × 控件）结果都必须登记一次；登记满 待完成 份后，
    由最后一份的登记方负责汇总预测并唤醒提交者。
    """

    待完成: int                                   # 剩余待登记的份数 = 任务数 × 控件数
    结果: List[List[dict]]                        # 按任务序号收集区域
    预测: Optional[List[dict]] = None             # 汇总后的 label studio 预测
    完成事件: threading.Event = field(default_factory=threading.Event)
    锁: threading.Lock = field(default_factory=threading.Lock)

    def 登记完成(self, 任务序号: int, 区域列表: List[dict]) -> bool:
        """登记一份结果；返回是否为最后一份（最后一份由调用方负责汇总并唤醒提交者）"""
        with self.锁:
            self.结果[任务序号].extend(区域列表)
            self.待完成 -= 1
            return self.待完成 <= 0


@dataclass
class 流水线任务:
    """predict() → 接收队列"""

    任务列表: List[Dict]
    控件列表: List[控件元数据]
    批次: 批次状态


@dataclass
class 推理输入:
    """接收线程 → 推理队列（同一图片字段的控件合并为一次推理；图片路径已确认有效）"""

    任务序号: int
    图片路径: str
    控件列表: List[控件元数据]
    批次: 批次状态


@dataclass
class 推理输出:
    """推理线程 → 后处理队列（每个控件一条；只投递有效的单图结果）"""

    任务序号: int
    结果: 单图结果
    控件: 控件元数据
    批次: 批次状态


@dataclass
class 后处理输出:
    """后处理线程 → 返回队列"""

    任务序号: int
    区域列表: List[dict]
    批次: 批次状态


# ---------------------------------------------------------------------------
# 三、后处理器：函数名与函数用 getattr 关联
# ---------------------------------------------------------------------------
class 后处理器:
    """后处理函数集合

    约定：函数名以 _后处理 结尾，签名统一为 (单图结果, 控件元数据) -> List[区域 dict]；
    输入是推理线程转好的纯 numpy 单图结果（不携带 torch 张量），
    输出为 label studio 固定结果格式（与官方 examples/yolo 保持一致）。
    实际调用哪个函数由 后处理配置.函数映射 里的字符串决定。
    """

    def 存在(self, 函数名: str) -> bool:
        """配置里的函数名是否有对应的后处理函数"""
        return callable(getattr(self, 函数名, None))

    def 应用(self, 函数名: str, 结果: 单图结果, 控件: 控件元数据) -> List[dict]:
        """配置里的 str 在这里 getattr 出内部函数并调用；只处理筛选后的有效结果"""
        函数 = getattr(self, 函数名)
        return 函数(结果, 控件)

    def 框_后处理(self, 结果: 单图结果, 控件: 控件元数据) -> List[dict]:
        """矩形框（RectangleLabels）后处理：归一化 xywh → 百分比"""
        区域列表 = []
        for 下标 in range(结果.检测数):
            标注名 = self._匹配标注(结果, 下标, 控件)
            if 标注名 is None:
                continue
            x, y, 宽, 高 = (float(值) for 值 in 结果.归一化框[下标])
            区域列表.append({
                "id": uuid4().hex[:9],
                "from_name": 控件.控件名,
                "to_name": 控件.目标名,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [标注名],
                    "x": (x - 宽 / 2) * 100,
                    "y": (y - 高 / 2) * 100,
                    "width": 宽 * 100,
                    "height": 高 * 100,
                },
                "score": float(结果.置信度[下标]),
            })
        return 区域列表

    def 掩码_后处理(self, 结果: 单图结果, 控件: 控件元数据) -> List[dict]:
        """掩码（BrushLabels）后处理：二值化后用 label studio 官方 mask2rle 编码"""
        if 结果.掩码 is None or len(结果.掩码) == 0:
            return []
        高, 宽 = 结果.原图高, 结果.原图宽
        区域列表 = []
        for 下标 in range(结果.检测数):
            标注名 = self._匹配标注(结果, 下标, 控件)
            if 标注名 is None:
                continue
            掩码 = (cv2.resize(结果.掩码[下标], (宽, 高)) > 0).astype("uint8") * 255
            区域列表.append({
                "id": uuid4().hex[:9],
                "from_name": 控件.控件名,
                "to_name": 控件.目标名,
                "original_width": 宽,
                "original_height": 高,
                "image_rotation": 0,
                "type": "brushlabels",
                "value": {
                    "format": "rle",
                    "rle": mask2rle(掩码),
                    "brushlabels": [标注名],
                },
                "score": float(结果.置信度[下标]),
            })
        return 区域列表

    def 多边形_后处理(self, 结果: 单图结果, 控件: 控件元数据) -> List[dict]:
        """多边形（PolygonLabels）后处理：掩码轮廓点集，归一化 → 百分比"""
        if 结果.多边形 is None or len(结果.多边形) == 0:
            return []
        区域列表 = []
        for 下标 in range(结果.检测数):
            标注名 = self._匹配标注(结果, 下标, 控件)
            if 标注名 is None:
                continue
            区域列表.append({
                "id": uuid4().hex[:9],
                "from_name": 控件.控件名,
                "to_name": 控件.目标名,
                "type": "polygonlabels",
                "value": {
                    "polygonlabels": [标注名],
                    "points": (结果.多边形[下标] * 100).tolist(),
                    "closed": True,
                },
                "score": float(结果.置信度[下标]),
            })
        return 区域列表

    @staticmethod
    def _匹配标注(结果: 单图结果, 下标: int, 控件: 控件元数据) -> Optional[str]:
        """模型类别名 → XML 标签映射；没有对应标注的检测直接丢弃"""
        类别名 = 结果.类别名.get(int(结果.类别下标[下标]))
        if 类别名 is None:
            return None
        return 控件.标签映射.get(类别名)


# ---------------------------------------------------------------------------
# 四、推理函数（唯一调用 YOLO 的位置）
# ---------------------------------------------------------------------------
_模型缓存: Dict[str, YOLO] = {}
_模型锁 = threading.Lock()


def 获取模型(路径: str) -> YOLO:
    """按路径缓存 YOLO 模型：进程内只加载一次，常驻复用"""
    with _模型锁:
        if 路径 not in _模型缓存:
            logger.info(f"加载 YOLO 模型：{路径}")
            _模型缓存[路径] = YOLO(路径)
        return _模型缓存[路径]


def 模型类别名(路径: Optional[str] = None) -> List[str]:
    """模型能识别的类别名列表（用于和 XML 配置里的标签做映射）"""
    return list(获取模型(路径 or 推理参数_默认.模型路径).names.values())


def 推理(图片路径: str, 参数: Optional[推理参数] = None):
    """对单张图片执行 YOLO 推理（单张图进、单张图出），返回 ultralytics 的 Results

    不开放参数 device=cpu、imgsz=960 写死在下面 model.predict 调用内部；
    其余开放参数全部来自 推理参数，未知参数可用 额外参数 透传。
    返回的 Results 紧接着要被 转numpy() 转成纯 numpy，不在线程间传递张量。
    """
    参数 = 参数 or 推理参数_默认
    模型 = 获取模型(参数.模型路径)
    结果集 = 模型.predict(
        source=图片路径,
        device="cpu",            # 不开放：固定使用 CPU
        imgsz=960,               # 不开放：固定 960
        conf=参数.置信度,
        iou=参数.IOU阈值,
        max_det=参数.最大检测数,
        classes=参数.类别索引,
        agnostic_nms=参数.类别无关NMS,
        verbose=False,
        **参数.额外参数,
    )
    return 结果集[0]


def 转numpy(结果) -> 单图结果:
    """ultralytics Results → 单图结果：只用 numpy 接口转换

    device=cpu 时框 / 掩码都是 CPU 张量，np.asarray 直接按缓冲区读取，
    不使用 .numpy()、张量 .tolist() 这类 torch 专有接口；转换后
    线程间传递的全部是纯 numpy 数组。
    """
    高, 宽 = 结果.orig_shape[:2]
    框 = getattr(结果, "boxes", None)
    if 框 is None:
        类别下标 = np.zeros(0, dtype=int)
        置信度 = np.zeros(0, dtype=float)
        归一化框 = np.zeros((0, 4), dtype=float)
    else:
        类别下标 = np.asarray(框.cls).astype(int).reshape(-1)
        置信度 = np.asarray(框.conf, dtype=float).reshape(-1)
        归一化框 = np.asarray(框.xywhn, dtype=float).reshape(-1, 4)

    掩码 = None
    多边形 = None
    掩码集 = getattr(结果, "masks", None)
    if 掩码集 is not None:
        掩码数据 = getattr(掩码集, "data", None)
        if 掩码数据 is not None and len(掩码数据) > 0:
            掩码 = np.asarray(掩码数据)
        点集列表 = getattr(掩码集, "xyn", None)
        if 点集列表:
            多边形 = [np.asarray(点集) for 点集 in 点集列表]

    return 单图结果(
        类别下标=类别下标,
        置信度=置信度,
        归一化框=归一化框,
        掩码=掩码,
        多边形=多边形,
        原图高=int(高),
        原图宽=int(宽),
        类别名=dict(getattr(结果, "names", None) or {}),
    )


def 解析图片路径(任务: Dict, 字段名: str) -> Optional[str]:
    """把 task['data'][字段] 解析成本地图片路径；失败返回 None（该组按空结果处理）"""
    取值 = (任务.get("data") or {}).get(字段名)
    if not 取值:
        logger.warning(f"任务 {任务.get('id')} 的 data 里缺少字段 {字段名}")
        return None
    # 本地文件先走快路径：get_local_path 不认普通本地路径（含 Windows 反斜杠绝对路径），
    # 会把它当 URL 去下载，所以这里先判断文件是否已存在
    if isinstance(取值, str) and os.path.exists(取值):
        return 取值
    try:
        return get_local_path(
            取值,
            hostname=os.environ.get("LABEL_STUDIO_URL"),
            access_token=os.environ.get("LABEL_STUDIO_API_KEY"),
            task_id=任务.get("id"),
        )
    except Exception:
        logger.exception(f"任务 {任务.get('id')} 图片下载/定位失败：{取值}")
        return None


# ---------------------------------------------------------------------------
# 五、进程级常驻流水线
# ---------------------------------------------------------------------------
class 推理流水线:
    """接收×1 → 推理×1 → 后处理×1-2 → 返回×1（固定线程池，总数不超过 max_thread=6）

    - 所有队列传递的都是 dataclass，每个单元只对应一张图（单张图进、单张图出）；
    - 推理天然串行（同一时刻只跑一张图），拆进流水线只是让各阶段解耦；
    - 每级先筛选再传递：拿不到图片 / 推理失败 / 无检测的单元就地按空结果登记，
      只有有效数据才会进入下一个线程；
    - YOLO 模型只在预热/推理线程里加载一次，常驻复用（进程级模型缓存）；
    - 工作线程是 daemon：解释器退出时不会因 join 常驻线程而卡死。
    """

    def __init__(
        self,
        参数: Optional[推理参数] = None,
        配置: Optional[后处理配置] = None,
        后处理线程数限制: int = 后处理线程数,
        超时: float = 流水线超时,
    ):
        self.参数 = 参数 or 推理参数_默认
        self.配置 = 配置 or 后处理配置_默认
        self.后处理器 = 后处理器()
        self.超时 = 超时
        self._接收队列: Queue = Queue()
        self._推理队列: Queue = Queue()
        self._后处理队列: Queue = Queue()
        self._返回队列: Queue = Queue()
        self._停止 = threading.Event()
        self._线程池: List[threading.Thread] = []
        self._启动线程(self._接收循环, "接收")
        self._启动线程(self._推理循环, "推理")
        for 编号 in range(max(1, min(2, 后处理线程数限制))):
            self._启动线程(self._后处理循环, f"后处理{编号 + 1}", 编号)
        self._启动线程(self._返回循环, "返回")
        # 启动即后台预热模型（一次性线程），首个请求不用等权重加载
        self._启动线程(self._预热模型, "预热")

    def _启动线程(self, 目标, 名称: str, *参数) -> None:
        线程 = threading.Thread(
            target=目标, args=参数, name=f"推理流水线-{名称}", daemon=True)
        线程.start()
        self._线程池.append(线程)

    def 提交(self, 任务: 流水线任务) -> List[dict]:
        """predict() 入口：投递任务后阻塞等待返回线程汇总结果"""
        if 任务.批次.待完成 <= 0:
            return []
        self._接收队列.put(任务)
        if not 任务.批次.完成事件.wait(self.超时):
            raise TimeoutError(f"推理流水线 {self.超时} 秒内未返回结果")
        return 任务.批次.预测 or []

    def 关闭(self) -> None:
        """停止流水线（atexit 自动调用）"""
        self._停止.set()
        for 线程 in self._线程池:
            线程.join(timeout=1.0)

    # ---- 各阶段 ----

    def _取(self, 队列: Queue):
        try:
            return 队列.get(timeout=0.5)
        except Empty:
            return None  # 只作空标记；流水线不投递 None 负载

    def _预热模型(self) -> None:
        try:
            模型类别名(self.参数.模型路径)
        except Exception:
            logger.exception("YOLO 模型预热失败，将在首次推理时重试")

    def _接收循环(self) -> None:
        while not self._停止.is_set():
            任务 = self._取(self._接收队列)
            if 任务 is None:
                continue
            try:
                self._分发任务(任务)
            except Exception:
                logger.exception("接收阶段处理失败")

    def _分发任务(self, 任务: 流水线任务) -> None:
        """按图片字段分组解析图片，同字段的多个控件合并成一次推理

        筛选：解析失败（拿不到图片）的组不进入推理线程，就地按空结果登记，
        保证批次计数仍然完整。
        """
        for 任务序号, 单个任务 in enumerate(任务.任务列表):
            按字段分组: Dict[str, List[控件元数据]] = {}
            for 控件 in 任务.控件列表:
                按字段分组.setdefault(控件.图片字段, []).append(控件)
            for 字段, 控件列表 in 按字段分组.items():
                图片路径 = 解析图片路径(单个任务, 字段)
                if 图片路径 is None:
                    for _ in 控件列表:
                        self._完成一份(任务.批次, 任务序号, [])
                    continue
                self._推理队列.put(推理输入(
                    任务序号=任务序号,
                    图片路径=图片路径,
                    控件列表=控件列表,
                    批次=任务.批次,
                ))

    def _推理循环(self) -> None:
        while not self._停止.is_set():
            输入 = self._取(self._推理队列)
            if 输入 is None:
                continue
            结果 = self._单张推理(输入)
            if 结果 is None or 结果.检测数 == 0:
                # 筛选：推理失败 / 没有检测的单元不传给后处理，就地按空结果登记
                for _ in 输入.控件列表:
                    self._完成一份(输入.批次, 输入.任务序号, [])
                continue
            # 一次单图推理服务同一张图上的多个控件：每个控件一条消息
            for 控件 in 输入.控件列表:
                self._后处理队列.put(推理输出(输入.任务序号, 结果, 控件, 输入.批次))

    def _单张推理(self, 输入: 推理输入) -> Optional[单图结果]:
        """单张图推理：YOLO 结果立刻转成纯 numpy 的单图结果；失败返回 None"""
        try:
            return 转numpy(推理(输入.图片路径, self.参数))
        except Exception:
            logger.exception(f"推理失败：{输入.图片路径}")
            return None

    def _后处理循环(self, 编号: int) -> None:
        while not self._停止.is_set():
            输出 = self._取(self._后处理队列)
            if 输出 is None:
                continue
            区域列表: List[dict] = []
            try:
                区域列表 = self.后处理器.应用(输出.控件.后处理函数名, 输出.结果, 输出.控件)
            except Exception:
                logger.exception(f"后处理失败：{输出.控件.后处理函数名}（线程 {编号}）")
            self._返回队列.put(后处理输出(输出.任务序号, 区域列表, 输出.批次))

    def _返回循环(self) -> None:
        while not self._停止.is_set():
            输出 = self._取(self._返回队列)
            if 输出 is None:
                continue
            self._完成一份(输出.批次, 输出.任务序号, 输出.区域列表)

    def _完成一份(self, 批次: 批次状态, 任务序号: int, 区域列表: List[dict]) -> None:
        """登记一份（任务 × 控件）结果；最后一份负责汇总预测并唤醒提交者"""
        if not 批次.登记完成(任务序号, 区域列表):
            return
        try:
            批次.预测 = self._组装预测(批次)
        except Exception:
            logger.exception("汇总预测结果失败")
            批次.预测 = 批次.预测 or []
        finally:
            批次.完成事件.set()

    @staticmethod
    def _组装预测(批次: 批次状态) -> List[dict]:
        """汇总成 label studio predict 响应格式（一条预测对应一个 task）"""
        预测列表 = []
        for 区域列表 in 批次.结果:
            分数列表 = [区域.get("score", 0.0) for 区域 in 区域列表]
            预测列表.append({
                "score": round(sum(分数列表) / len(分数列表), 4) if 分数列表 else 0.0,
                "result": 区域列表,
            })
        return 预测列表


_流水线单例: Optional[推理流水线] = None
_流水线锁 = threading.Lock()


def 获取流水线() -> 推理流水线:
    """进程级常驻流水线单例：所有 /predict 请求共用一份"""
    global _流水线单例
    with _流水线锁:
        if _流水线单例 is None:
            _流水线单例 = 推理流水线()
            atexit.register(_流水线单例.关闭)
        return _流水线单例
