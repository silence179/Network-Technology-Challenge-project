"""
Visualization utilities: font setup and result plotting.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_font():
    """Try to set a Chinese-capable font, fall back to DejaVu Sans."""
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans', 'Arial']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.family'] = [font, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return


setup_font()
