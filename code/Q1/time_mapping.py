"""问题1自然日时间映射；标签和数据同时校验，禁止仅凭行号判定一致。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from microgrid_common import clock_label, end_minutes


def canonical_periods(ends):
    ends = [int(v) for v in ends]
    if ends != list(range(10, 1441, 10)):
        raise ValueError('问题1必须覆盖当天0:00至24:00的144个连续十分钟区间')
    return [f'{clock_label(end-10)}-{clock_label(end)}' for end in ends]


def period_bounds(label):
    start, end = str(label).split('-')
    return end_minutes(start), end_minutes(end)


def validate_export_periods(labels, ends):
    expected = [period_bounds(v) for v in canonical_periods(ends)]
    actual = [period_bounds(v) for v in labels]
    if actual != expected:
        raise ValueError('Excel时间标签与模型自然日区间不一致，存在偏移、遗漏或顺序错误')
    return True
