"""生成问题1预处理阶段的三组输入数据日内特征图，不读取或绘制优化结果。"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'.matplotlib'))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from microgrid_common import read_single_day
from time_mapping import canonical_periods


def _font():
    path = Path('/System/Library/Fonts/STHeiti Medium.ttc')
    return font_manager.FontProperties(fname=str(path)) if path.exists() else None


def _clock(index):
    minutes = index*10
    return f'{minutes//60:02d}:{minutes%60:02d}'


def input_statistics(data):
    price = np.asarray(data['price'])
    load_kw = np.asarray(data['load'])*6
    pv_kw = np.asarray(data['pv'])*6
    return {
        'record_count': len(price),
        'period_start': canonical_periods(data['end_minutes'])[0],
        'period_end': canonical_periods(data['end_minutes'])[-1],
        'price_min_yuan_per_kwh': float(price.min()),
        'price_min_period': canonical_periods(data['end_minutes'])[int(price.argmin())],
        'price_max_yuan_per_kwh': float(price.max()),
        'price_max_period': canonical_periods(data['end_minutes'])[int(price.argmax())],
        'load_max_kw': float(load_kw.max()),
        'load_max_period': canonical_periods(data['end_minutes'])[int(load_kw.argmax())],
        'pv_max_kw': float(pv_kw.max()),
        'pv_max_period': canonical_periods(data['end_minutes'])[int(pv_kw.argmax())],
    }


def plot_input_profiles(data, output_dir=OUT):
    """只接收附件1预处理后的输入字段，输出PNG、SVG和描述统计JSON。"""
    output_dir = Path(output_dir)
    price = np.asarray(data['price'])
    load_kw = np.asarray(data['load'])*6
    pv_kw = np.asarray(data['pv'])*6
    ends = np.asarray(data['end_minutes'])
    if len(price) != 144 or not (len(load_kw) == len(pv_kw) == len(ends)):
        raise ValueError('输入图必须使用完整的144个十分钟时段')
    x = (ends-5)/60  # 每个十分钟区间的中心
    cn = _font()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10})
    colors = ['#334155', '#d99a13', '#be185d']
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 8.2), sharex=True,
                             gridspec_kw={'hspace': .13})
    series = [
        (load_kw, '小区负荷', '功率（kW）', colors[0]),
        (pv_kw, '光伏预测', '功率（kW）', colors[1]),
        (price, '外网电价', '电价（元/kWh）', colors[2]),
    ]
    for ax, (values, label, ylabel, color) in zip(axes, series):
        ax.plot(x, values, color=color, linewidth=1.8, label=label)
        ax.fill_between(x, values, color=color, alpha=.12)
        ax.set_ylabel(ylabel, fontproperties=cn)
        ax.legend(loc='upper right', frameon=False, prop=cn)
        ax.grid(axis='both', alpha=.2, linewidth=.7)
        ax.set_xlim(0, 24)
        for hour in (6, 12, 18):
            ax.axvline(hour, color='#94a3b8', linestyle='--', linewidth=.8, alpha=.75)
    annotations = [
        (int(load_kw.argmax()), f'日最大负荷 {load_kw.max():.1f} kW'),
        (int(pv_kw.argmax()), f'光伏峰值 {pv_kw.max():.1f} kW'),
        (int(price.argmin()), f'最低电价 {price.min():.4f} 元/kWh'),
    ]
    for ax, values, (index, label), color in zip(
            axes, [load_kw, pv_kw, price], annotations, colors):
        ax.scatter([x[index]], [values[index]], color=color, s=28, zorder=3)
        ax.annotate(label, (x[index], values[index]), xytext=(8, 10),
                    textcoords='offset points', fontproperties=cn, color=color,
                    arrowprops={'arrowstyle':'-', 'color':color, 'lw':.8})
    axes[-1].set_xticks(np.arange(0, 25, 2))
    axes[-1].set_xlabel('自然日时刻', fontproperties=cn)
    fig.suptitle('问题1三组输入数据的日内变化特征', fontproperties=cn, fontsize=16)
    fig.text(.5, .01, '虚线标示 6:00、12:00 和 18:00；曲线仅来自附件1输入数据',
             ha='center', fontproperties=cn, fontsize=9, color='#475569')
    fig.subplots_adjust(left=.10, right=.98, top=.92, bottom=.08)
    fig.savefig(output_dir/'input_daily_patterns.png', dpi=220)
    fig.savefig(output_dir/'input_daily_patterns.svg')
    plt.close(fig)
    stats = input_statistics(data)
    (output_dir/'input_statistics.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return stats


if __name__ == '__main__':
    result = plot_input_profiles(read_single_day(ROOT/'附件/附件1.xlsx'))
    print(json.dumps(result, ensure_ascii=False, indent=2))
