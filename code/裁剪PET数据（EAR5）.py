import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray  # 只需要导入即可为 xarray 对象增加 .rio 空间处理方法。
import xarray as xr


# =============================================================================
# 一、基础路径参数
# =============================================================================
# PET 原始 NetCDF 文件所在目录。
# 修改意义：
# - 如果以后更换 PET 数据目录，只需要改这一项；
# - 脚本会在这个目录下自动寻找形如 PET_202501_hourly.nc 的文件；
# - 2025 年 10、11、12 月文件名允许写成 PET_2025010_hourly.nc、
#   PET_2025011_hourly.nc、PET_2025012_hourly.nc。
PET原始数据目录 = Path(r"E:\forest_microclimate\ForestMicroclimate\PET_TerraClimate")

# 福建省行政边界 shp 文件路径。
# 修改意义：
# - 该文件用于提供裁剪范围；
# - 如果以后更换为其他省份或研究区边界，只需要改这一项。
福建省边界文件 = Path(r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp")

# 输出文件夹名称。
# 修改意义：
# - 该文件夹会自动创建在 PET 原始数据目录下；
# - 按你的要求，这里固定命名为 fujian_PET；
# - 后续所有裁剪结果、空间分布图和统计表都保存在这个文件夹内。
输出文件夹名称 = "fujian_PET"


# =============================================================================
# 二、数据变量与单位参数
# =============================================================================
# PET 变量名。
# 修改意义：
# - ERA5-Land 潜在蒸发常见变量名是 pev；
# - 如果你的 NetCDF 中变量名不是 pev，可以改为实际变量名；
# - 如果设置为 None，脚本会优先自动查找 pev、pet、potential_evaporation。
PET变量名 = "pev"

# PET 单位换算系数。
# 修改意义：
# - ERA5-Land 的 pev 常见单位是 m，乘以 1000 后变为 mm；
# - 如果你的数据已经是 mm，可改为 1。
米转毫米系数 = 1000.0

# 是否将 ERA5-Land 的负向蒸发值转为正向 PET。
# 修改意义：
# - ERA5-Land 蒸发相关变量通常以负值表示从地表向大气的通量；
# - True 表示使用 PET(mm) = -pev(m) * 1000，这是常见处理方式；
# - 如果你的数据已经是正值 PET，可改为 False。
负值转正 = True

# 是否把换算后仍小于 0 的 PET 值修正为 0。
# 修改意义：
# - 用于避免极少量异常符号值影响月累计；
# - True 表示负值修正为 0；
# - False 表示保留原始换算后的数值。
PET最小值限制为0 = True


# =============================================================================
# 三、输出与绘图参数
# =============================================================================
# NetCDF 输出压缩开关。
# 修改意义：
# - True：输出文件更小，但写出速度可能更慢；
# - False：输出更快，但文件体积可能更大。
启用NetCDF压缩 = True

# NetCDF 压缩等级。
# 修改意义：
# - 取值通常为 1 到 9；
# - 数值越大压缩率越高，但写出越慢；
# - 只在 启用NetCDF压缩 = True 时生效。
NetCDF压缩等级 = 4

# 进度条宽度。
# 修改意义：
# - 数值越大，控制台进度条越长；
# - 只影响显示效果，不影响计算结果。
进度条宽度 = 36

# 空间分布图尺寸，单位为英寸。
# 修改意义：
# - 图像越大，地图细节和文字越清晰；
# - 文件体积也会相应增大。
图片宽度英寸 = 10
图片高度英寸 = 8

# 空间分布图分辨率，单位为 DPI。
# 修改意义：
# - DPI 越高，导出的 PNG 越清晰；
# - DPI 越高，图片文件通常越大。
图片DPI = 300

# PET 空间分布图配色方案。
# 修改意义：
# - 常用可选值包括 YlOrRd、YlGnBu、viridis、turbo；
# - 只影响图片颜色，不影响裁剪数据和统计表。
PET配色方案 = "YlOrRd"

# 色带范围。
# 修改意义：
# - None 表示每张图根据当月数据自动确定颜色范围；
# - 如果希望不同月份之间颜色可直接对比，可改成固定数值，例如 0 和 250。
色带最小值 = None
色带最大值 = None

# 福建省边界线样式。
# 修改意义：
# - 用于在空间分布图上叠加行政边界；
# - 只影响图片显示，不影响裁剪数据。
边界线颜色 = "black"
边界线宽度 = 0.8


def 打印分隔线():
    """打印分隔线，用于区分不同处理阶段。"""
    print("\n" + "=" * 80)


def 打印进度条(标题, 当前数量, 总数量):
    """
    打印控制台进度条。

    参数说明：
    - 标题：当前进度条代表的处理内容，例如“文件处理进度”；
    - 当前数量：已经完成或正在执行到第几个任务，从 0 或 1 开始均可；
    - 总数量：任务总数，用于计算百分比。

    设计目的：
    - 不依赖 tqdm 等额外进度条库；
    - 在每个关键步骤和每个文件处理节点都能看到进度。
    """
    if 总数量 <= 0:
        完成比例 = 0
    else:
        完成比例 = min(max(当前数量 / 总数量, 0), 1)

    已完成宽度 = int(完成比例 * 进度条宽度)
    进度条 = "█" * 已完成宽度 + "-" * (进度条宽度 - 已完成宽度)
    print(f"[{标题}] |{进度条}| {完成比例:>6.1%} ({当前数量}/{总数量})")


def 配置中文字体():
    """
    配置 Matplotlib 中文显示。

    参数说明：
    - font.sans-serif 按优先级列出常见中文字体；
    - axes.unicode_minus = False 用于避免坐标轴负号显示为方块。

    注意：
    - 这里不会安装任何字体或依赖；
    - 如果系统缺少某个字体，Matplotlib 会自动尝试下一个字体。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def 解析PET文件日期(nc文件路径):
    """
    从 PET 文件名中解析年份和月份。

    参数说明：
    - nc文件路径：单个 PET NetCDF 文件路径；
    - 文件名应类似 PET_202501_hourly.nc；
    - 2025 年 10、11、12 月兼容 PET_2025010_hourly.nc 等命名。

    返回值：
    - (年份, 月份)，例如 (2025, 1) 或 (2025, 10)。
    """
    匹配结果 = re.match(r"^PET_(\d{4})(\d{2,3})_hourly\.nc$", nc文件路径.name)
    if not 匹配结果:
        raise ValueError(f"文件名不符合 PET_年月_hourly.nc 规则：{nc文件路径.name}")

    年份 = int(匹配结果.group(1))
    月份 = int(匹配结果.group(2))
    if 月份 < 1 or 月份 > 12:
        raise ValueError(f"文件名中的月份不在 1-12 范围内：{nc文件路径.name}")

    return 年份, 月份


def 查找PET文件列表():
    """
    查找并排序所有待处理 PET 文件。

    处理逻辑：
    - 只读取 PET原始数据目录 下以 PET_ 开头、以 _hourly.nc 结尾的文件；
    - 解析文件名中的年份和月份；
    - 按年份、月份从小到大排序，保证处理顺序稳定。
    """
    候选文件列表 = list(PET原始数据目录.glob("PET_*_hourly.nc"))
    文件信息列表 = []

    for nc文件路径 in 候选文件列表:
        try:
            年份, 月份 = 解析PET文件日期(nc文件路径)
        except ValueError as 错误:
            print(f"跳过无法识别的文件：{错误}")
            continue
        文件信息列表.append((年份, 月份, nc文件路径))

    文件信息列表.sort(key=lambda 项: (项[0], 项[1], 项[2].name))
    return 文件信息列表


def 校验输入路径并创建输出目录():
    """
    校验输入数据是否存在，并创建输出目录。

    关键步骤说明：
    - 先检查 PET 原始数据目录，避免路径错误导致后续批处理失败；
    - 再检查福建省 shp 文件，确保裁剪边界存在；
    - 最后创建 fujian_PET 输出目录，用于集中保存所有结果。
    """
    if not PET原始数据目录.exists():
        raise FileNotFoundError(f"未找到 PET 原始数据目录：{PET原始数据目录}")

    if not 福建省边界文件.exists():
        raise FileNotFoundError(f"未找到福建省边界文件：{福建省边界文件}")

    输出目录 = PET原始数据目录 / 输出文件夹名称
    输出目录.mkdir(parents=True, exist_ok=True)
    return 输出目录


def 读取福建省边界():
    """
    读取福建省行政边界，并统一到 EPSG:4326 坐标系。

    参数意义：
    - ERA5-Land 通常使用经纬度坐标，也就是 EPSG:4326；
    - 裁剪前统一坐标系，可以避免边界和栅格错位。
    """
    福建省边界 = gpd.read_file(福建省边界文件)

    if 福建省边界.crs is None:
        raise ValueError("福建省 shp 文件没有坐标系信息，无法安全裁剪。请先检查 shp 的 .prj 文件。")

    if 福建省边界.crs.to_epsg() != 4326:
        福建省边界 = 福建省边界.to_crs(epsg=4326)
        print("福建省边界已重投影到 EPSG:4326。")
    else:
        print("福建省边界已经是 EPSG:4326。")

    return 福建省边界


def 识别PET变量名(数据集):
    """
    识别 NetCDF 中的 PET 变量名。

    参数说明：
    - 数据集：xarray.open_dataset 读取后的 Dataset；
    - 如果脚本顶部 PET变量名 不为 None，则优先使用该变量；
    - 否则按常见变量名自动查找。
    """
    if PET变量名 is not None:
        if PET变量名 not in 数据集.data_vars:
            raise KeyError(f"未找到指定 PET 变量：{PET变量名}；可用变量为：{list(数据集.data_vars)}")
        return PET变量名

    for 候选变量名 in ["pev", "pet", "potential_evaporation", "potential_evapotranspiration"]:
        if 候选变量名 in 数据集.data_vars:
            return 候选变量名

    raise KeyError(f"未自动识别 PET 变量；可用变量为：{list(数据集.data_vars)}")


def 识别空间维度(数据数组):
    """
    识别经度和纬度维度名称。

    参数说明：
    - 数据数组：待裁剪的 PET DataArray；
    - ERA5-Land 常见维度名为 longitude 和 latitude；
    - 有些文件可能使用 lon/lat 或 x/y，因此这里做兼容识别。
    """
    经度候选 = ["longitude", "lon", "x"]
    纬度候选 = ["latitude", "lat", "y"]

    经度维度 = next((维度 for 维度 in 经度候选 if 维度 in 数据数组.dims or 维度 in 数据数组.coords), None)
    纬度维度 = next((维度 for 维度 in 纬度候选 if 维度 in 数据数组.dims or 维度 in 数据数组.coords), None)

    if 经度维度 is None or 纬度维度 is None:
        raise KeyError(f"无法识别经纬度维度；当前维度为：{list(数据数组.dims)}")

    return 经度维度, 纬度维度


def 识别时间维度(数据数组):
    """
    识别时间维度名称。

    参数说明：
    - ERA5-Land 常见时间维度为 time 或 valid_time；
    - 月累计 PET 需要沿时间维求和，因此必须识别时间维。
    """
    时间候选 = ["time", "valid_time", "datetime", "date"]
    时间维度 = next((维度 for 维度 in 时间候选 if 维度 in 数据数组.dims), None)

    if 时间维度 is None:
        raise KeyError(f"无法识别时间维度；当前维度为：{list(数据数组.dims)}")

    return 时间维度


def 转换为正向毫米PET(PET原始数据):
    """
    将原始 PET 数据转换为正向毫米单位。

    处理步骤：
    - 如果 负值转正 = True，使用 -PET * 米转毫米系数；
    - 如果 负值转正 = False，使用 PET * 米转毫米系数；
    - 如果 PET最小值限制为0 = True，把换算后小于 0 的值设为 0。

    返回值：
    - 单位为 mm 的小时 PET 数据。
    """
    if 负值转正:
        PET毫米 = -PET原始数据 * 米转毫米系数
    else:
        PET毫米 = PET原始数据 * 米转毫米系数

    if PET最小值限制为0:
        PET毫米 = PET毫米.where(PET毫米 >= 0, 0)

    PET毫米.name = "PET_mm"
    PET毫米.attrs["long_name"] = "正向潜在蒸散发"
    PET毫米.attrs["units"] = "mm"
    PET毫米.attrs["说明"] = "由原始 ERA5-Land PET 按脚本参数转换得到；默认处理为 -pev * 1000。"
    return PET毫米


def 计算月累计PET(PET小时数据):
    """
    沿时间维度求和，得到月累计 PET 空间栅格。

    参数说明：
    - PET小时数据：已经裁剪并转换为 mm 的小时 PET；
    - skipna=True 表示忽略缺失值求和。
    """
    时间维度 = 识别时间维度(PET小时数据)
    月累计PET = PET小时数据.sum(dim=时间维度, skipna=True)
    月累计PET.name = "PET_monthly_total_mm"
    月累计PET.attrs["long_name"] = "月累计潜在蒸散发"
    月累计PET.attrs["units"] = "mm"
    return 月累计PET


def 生成NetCDF编码参数(输出数据集):
    """
    生成 NetCDF 写出编码参数。

    参数说明：
    - 输出数据集：准备保存的 xarray Dataset；
    - 当启用压缩时，为每个数据变量设置 zlib 和 complevel；
    - 不启用压缩时返回 None，让 xarray 使用默认写出方式。
    """
    if not 启用NetCDF压缩:
        return None

    return {
        变量名: {"zlib": True, "complevel": NetCDF压缩等级}
        for 变量名 in 输出数据集.data_vars
    }


def 绘制并保存月累计PET空间分布图(月累计PET, 福建省边界, 年份, 月份, 输出图片路径):
    """
    绘制单月累计 PET 空间分布图并保存为 PNG。

    参数说明：
    - 月累计PET：二维栅格数据，单位为 mm；
    - 福建省边界：GeoDataFrame，用于叠加行政边界线；
    - 年份、月份：用于生成中文标题；
    - 输出图片路径：PNG 保存位置。
    """
    配置中文字体()

    图形, 坐标轴 = plt.subplots(figsize=(图片宽度英寸, 图片高度英寸))
    图形.patch.set_facecolor("white")
    坐标轴.set_facecolor("white")

    绘图对象 = 月累计PET.plot(
        ax=坐标轴,
        cmap=PET配色方案,
        vmin=色带最小值,
        vmax=色带最大值,
        add_colorbar=True,
        cbar_kwargs={
            "label": "月累计 PET（mm）",
            "shrink": 0.86,
            "pad": 0.02,
        },
    )

    福建省边界.boundary.plot(
        ax=坐标轴,
        color=边界线颜色,
        linewidth=边界线宽度,
    )

    坐标轴.set_title(f"福建省{年份}年{月份:02d}月累计PET空间分布图", fontsize=14, pad=12)
    坐标轴.set_xlabel("经度")
    坐标轴.set_ylabel("纬度")
    坐标轴.grid(False)

    if hasattr(绘图对象, "colorbar") and 绘图对象.colorbar is not None:
        绘图对象.colorbar.ax.tick_params(labelsize=9)
        绘图对象.colorbar.set_label("月累计 PET（mm）", fontsize=10)

    图形.tight_layout()
    图形.savefig(输出图片路径, dpi=图片DPI, bbox_inches="tight")
    plt.close(图形)


def 计算月统计(月累计PET):
    """
    计算福建省范围内月累计 PET 的空间统计值。

    统计项说明：
    - 平均值：福建省范围内所有有效栅格的月累计 PET 平均值；
    - 最大值：有效栅格中的最大月累计 PET；
    - 最小值：有效栅格中的最小月累计 PET；
    - 空间累计值：有效栅格月累计 PET 的直接求和；
    - 有效栅格数：参与统计的非空栅格数量。

    注意：
    - 空间累计值是栅格值求和，不是面积加权水量体积；
    - 如果后续需要严格体积统计，应增加栅格面积权重。
    """
    有效数据 = 月累计PET.where(月累计PET.notnull())
    有效栅格数 = int(有效数据.count().item())

    if 有效栅格数 == 0:
        return {
            "PET平均值_mm": None,
            "PET最大值_mm": None,
            "PET最小值_mm": None,
            "PET空间累计值_栅格求和_mm": None,
            "有效栅格数": 0,
        }

    return {
        "PET平均值_mm": round(float(有效数据.mean(skipna=True).item()), 6),
        "PET最大值_mm": round(float(有效数据.max(skipna=True).item()), 6),
        "PET最小值_mm": round(float(有效数据.min(skipna=True).item()), 6),
        "PET空间累计值_栅格求和_mm": round(float(有效数据.sum(skipna=True).item()), 6),
        "有效栅格数": 有效栅格数,
    }


def 处理单个PET文件(年份, 月份, nc文件路径, 福建省边界, 输出目录):
    """
    裁剪单个 PET 文件，生成 NetCDF、空间分布图和统计记录。

    核心流程：
    - 读取原始 NetCDF；
    - 识别 PET 变量、经纬度维度和时间维度；
    - 将 ERA5-Land PET 转为正向 mm；
    - 使用福建省边界裁剪；
    - 计算月累计 PET；
    - 保存英文命名的 NetCDF 和中文命名的 PNG；
    - 返回本月统计结果。
    """
    print(f"\n正在处理：{nc文件路径.name} -> {年份}年{月份:02d}月")
    文件标题 = f"福建省{年份}年{月份:02d}月PET"
    # NetCDF 文件使用英文命名，避免部分 Windows 环境中的 netCDF4/HDF5 后端
    # 在创建中文 .nc 文件名时误报 PermissionError。
    # 命名规则：在原始文件名前增加 fujian_ 前缀，例如
    # PET_202501_hourly.nc -> fujian_PET_202501_hourly.nc。
    输出nc路径 = 输出目录 / f"fujian_{nc文件路径.name}"
    输出图片路径 = 输出目录 / f"{文件标题}月累计空间分布图.png"

    with xr.open_dataset(nc文件路径) as 数据集:
        PET变量 = 识别PET变量名(数据集)
        PET原始数据 = 数据集[PET变量]
        经度维度, 纬度维度 = 识别空间维度(PET原始数据)
        时间维度 = 识别时间维度(PET原始数据)

        print(f"识别变量：{PET变量}")
        print(f"识别维度：时间={时间维度}，经度={经度维度}，纬度={纬度维度}")

        # 步骤 1：单位和符号转换。
        # 这里把 ERA5-Land 常见的负向米单位 pev 转换为正向毫米单位 PET，
        # 这样后续月累计、绘图和统计都使用更直观的 mm。
        PET毫米 = 转换为正向毫米PET(PET原始数据)

        # 步骤 2：写入空间参考。
        # rioxarray 裁剪需要知道哪两个维度是 x/y，以及数据使用的 CRS；
        # ERA5-Land 经纬度数据通常是 EPSG:4326。
        PET毫米 = PET毫米.rio.set_spatial_dims(x_dim=经度维度, y_dim=纬度维度)
        PET毫米 = PET毫米.rio.write_crs("EPSG:4326")

        # 步骤 3：按福建省行政边界裁剪。
        # drop=True 表示只保留福建省边界外包矩形范围内的数据；
        # all_touched=False 表示按默认规则判断栅格是否落入边界。
        PET裁剪后 = PET毫米.rio.clip(
            福建省边界.geometry,
            福建省边界.crs,
            drop=True,
            all_touched=False,
        )

        # 步骤 4：计算月累计 PET。
        # 原始文件是小时数据，因此沿时间维求和可得到每个栅格整月累计 PET。
        月累计PET = 计算月累计PET(PET裁剪后)

        # 步骤 5：组织输出 Dataset。
        # 同时保存小时 PET 和月累计 PET，便于后续既能复用裁剪后的小时数据，
        # 也能直接读取月尺度空间分布结果。
        输出数据集 = xr.Dataset(
            {
                "PET_mm": PET裁剪后,
                "PET_monthly_total_mm": 月累计PET,
            }
        )
        输出数据集.attrs["数据说明"] = "福建省范围内 ERA5-Land PET 裁剪结果。"
        输出数据集.attrs["单位处理"] = "默认使用 PET(mm) = -pev(m) * 1000。"
        输出数据集.attrs["裁剪范围"] = "福建省行政边界"
        输出数据集.attrs["对应原始文件"] = nc文件路径.name

        编码参数 = 生成NetCDF编码参数(输出数据集)

        print("保存裁剪后的 NetCDF 文件...")
        输出数据集.to_netcdf(输出nc路径, encoding=编码参数)

        print("绘制并保存月累计 PET 空间分布图...")
        绘制并保存月累计PET空间分布图(月累计PET, 福建省边界, 年份, 月份, 输出图片路径)

        月统计 = 计算月统计(月累计PET)

    月统计.update(
        {
            "年份": 年份,
            "月份": 月份,
            "原始文件": nc文件路径.name,
            "裁剪结果文件": 输出nc路径.name,
            "空间分布图文件": 输出图片路径.name,
        }
    )
    return 月统计


def 保存统计表(统计记录列表, 输出目录):
    """
    保存所有月份的 PET 统计表。

    参数说明：
    - 统计记录列表：每个月处理后返回的一行统计结果；
    - 输出目录：fujian_PET 目录。

    输出说明：
    - 文件名使用中文：福建省PET月统计表.csv；
    - 编码使用 utf-8-sig，便于 Excel 直接打开中文不乱码。
    """
    统计表路径 = 输出目录 / "福建省PET月统计表.csv"
    统计表 = pd.DataFrame(统计记录列表)

    列顺序 = [
        "年份",
        "月份",
        "PET平均值_mm",
        "PET最大值_mm",
        "PET最小值_mm",
        "PET空间累计值_栅格求和_mm",
        "有效栅格数",
        "原始文件",
        "裁剪结果文件",
        "空间分布图文件",
    ]
    统计表 = 统计表[列顺序]
    统计表.to_csv(统计表路径, index=False, encoding="utf-8-sig")
    return 统计表路径


def 主函数():
    """执行福建省 ERA5-Land PET 批量裁剪、月累计绘图和统计表生成流程。"""
    打印分隔线()
    print("开始执行：福建省 ERA5-Land PET 批量裁剪、月累计空间分布图和统计表生成")
    print(f"PET 原始数据目录：{PET原始数据目录}")
    print(f"福建省边界文件：{福建省边界文件}")

    总步骤数 = 5

    # 关键步骤 1：检查输入路径并创建输出目录。
    # 这样可以在正式批处理前尽早发现路径问题，避免处理到中途才失败。
    打印进度条("关键步骤", 1, 总步骤数)
    输出目录 = 校验输入路径并创建输出目录()
    print(f"输出目录：{输出目录}")

    # 关键步骤 2：查找并排序所有 PET 文件。
    # 排序后按月份顺序处理，输出图和统计表也会保持时间顺序。
    打印进度条("关键步骤", 2, 总步骤数)
    PET文件信息列表 = 查找PET文件列表()
    if not PET文件信息列表:
        raise FileNotFoundError(f"未在目录中找到 PET_*_hourly.nc 文件：{PET原始数据目录}")
    print(f"识别到 {len(PET文件信息列表)} 个 PET 文件。")

    # 关键步骤 3：读取福建省边界。
    # 边界只需读取一次，后续所有月份复用同一个 GeoDataFrame。
    打印进度条("关键步骤", 3, 总步骤数)
    福建省边界 = 读取福建省边界()
    print(f"福建省边界要素数量：{len(福建省边界)}")

    # 关键步骤 4：逐月裁剪、保存 NetCDF、绘图并计算统计值。
    # 每个文件都会额外显示文件级进度，便于观察批处理推进情况。
    打印进度条("关键步骤", 4, 总步骤数)
    统计记录列表 = []
    文件总数 = len(PET文件信息列表)

    for 序号, (年份, 月份, nc文件路径) in enumerate(PET文件信息列表, start=1):
        打印进度条("文件处理进度", 序号 - 1, 文件总数)
        月统计 = 处理单个PET文件(年份, 月份, nc文件路径, 福建省边界, 输出目录)
        统计记录列表.append(月统计)
        打印进度条("文件处理进度", 序号, 文件总数)

    # 关键步骤 5：保存中文统计表。
    # 统计表汇总所有月份，便于直接查看福建省范围内 PET 的月尺度变化。
    打印进度条("关键步骤", 5, 总步骤数)
    统计表路径 = 保存统计表(统计记录列表, 输出目录)

    打印分隔线()
    print("全部处理完成。")
    print(f"输出目录：{输出目录}")
    print(f"统计表：{统计表路径}")
    print("每个月均已输出：英文命名的裁剪 NetCDF 文件、中文命名的月累计 PET 空间分布图。")


if __name__ == "__main__":
    try:
        主函数()
    except Exception as 错误:
        打印分隔线()
        print("脚本执行失败，请根据以下错误检查路径、变量名、坐标系或 NetCDF 文件结构。")
        print(f"{type(错误).__name__}: {错误}")
        sys.exit(1)
