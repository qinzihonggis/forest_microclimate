import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rioxarray  # 只需要导入即可为 xarray 对象增加 .rio 空间处理方法。
import xarray as xr


# =============================================================================
# 一、基础路径参数
# =============================================================================
# TerraClimate PET 原始 NetCDF 文件路径。
# 修改意义：
# - 当前脚本按“单个年度文件”处理 TerraClimate PET；
# - 如果以后处理其他年份，只需要把这里改成对应年份的 TerraClimate_pet_YYYY.nc。
PET原始文件路径 = Path(
    r"E:\forest_microclimate\ForestMicroclimate\PET_TerraClimate\TerraClimate_pet_2002.nc"
)

# 福建省行政边界 shp 文件路径。
# 修改意义：
# - 该文件用于提供裁剪范围；
# - 其他输入数据路径按你的要求保持不变；
# - 如果以后更换研究区边界，只需要改这一项。
福建省边界文件 = Path(r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp")

# 输出文件夹名称。
# 修改意义：
# - 该文件夹会自动创建在 TerraClimate PET 原始文件所在目录下；
# - 按你的要求，这里固定命名为 fujian_PET_TerraClimate；
# - 裁剪后的 NetCDF 和统计表都会保存到该文件夹内。
输出文件夹名称 = "fujian_PET_TerraClimate"


# =============================================================================
# 二、数据变量与单位参数
# =============================================================================
# TerraClimate PET 变量名。
# 修改意义：
# - TerraClimate PET 文件中的变量名通常为 pet；
# - 如果后续文件变量名不同，可以在这里改为实际变量名。
PET变量名 = "pet"

# PET 单位名称。
# 修改意义：
# - 当前 TerraClimate 文件元数据中 pet 的单位已经是 mm；
# - 因此本脚本不再进行 ERA5-Land 那种 -pev * 1000 的符号和单位转换。
PET单位 = "mm"

# 是否把负值修正为 0。
# 修改意义：
# - TerraClimate PET 理论上应为非负值；
# - True 表示如果裁剪后存在少量负值，就修正为 0；
# - False 表示完全保留原始数值。
PET最小值限制为0 = True


# =============================================================================
# 三、输出参数
# =============================================================================
# 裁剪后年度 NetCDF 输出文件名前缀。
# 修改意义：
# - NetCDF 文件使用英文命名，避免部分 Windows 环境中的 netCDF4/HDF5 后端
#   在创建中文 .nc 文件名时误报 PermissionError；
# - 表格和图片仍按你的要求保持中文命名。
裁剪后NetCDF文件名前缀 = "fujian_"

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

def 打印分隔线():
    """打印分隔线，用于区分不同处理阶段。"""
    print("\n" + "=" * 80)


def 打印进度条(标题, 当前数量, 总数量):
    """
    打印控制台进度条。

    参数说明：
    - 标题：当前进度条代表的处理内容，例如“关键步骤”或“统计进度”；
    - 当前数量：已经完成或正在执行到第几个任务；
    - 总数量：任务总数，用于计算百分比。

    设计目的：
    - 不依赖 tqdm 等额外进度条库；
    - 在路径检查、裁剪、保存和统计表生成等关键步骤都能看到进度。
    """
    if 总数量 <= 0:
        完成比例 = 0
    else:
        完成比例 = min(max(当前数量 / 总数量, 0), 1)

    已完成宽度 = int(完成比例 * 进度条宽度)
    进度条 = "█" * 已完成宽度 + "-" * (进度条宽度 - 已完成宽度)
    print(f"[{标题}] |{进度条}| {完成比例:>6.1%} ({当前数量}/{总数量})")


def 校验输入路径并创建输出目录():
    """
    校验输入文件是否存在，并创建输出目录。

    关键步骤说明：
    - 先检查 TerraClimate PET 年度 NetCDF 文件，避免路径错误导致后续失败；
    - 再检查福建省 shp 文件，确保裁剪边界存在；
    - 最后在原始 PET 文件所在目录创建 fujian_PET_TerraClimate 输出目录。
    """
    if not PET原始文件路径.exists():
        raise FileNotFoundError(f"未找到 TerraClimate PET 文件：{PET原始文件路径}")

    if not 福建省边界文件.exists():
        raise FileNotFoundError(f"未找到福建省边界文件：{福建省边界文件}")

    输出目录 = PET原始文件路径.parent / 输出文件夹名称
    输出目录.mkdir(parents=True, exist_ok=True)
    return 输出目录


def 读取福建省边界():
    """
    读取福建省行政边界，并统一到 EPSG:4326 坐标系。

    参数意义：
    - TerraClimate 使用 WGS84 经纬度坐标，也就是 EPSG:4326；
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


def 识别空间维度(数据数组):
    """
    识别经度和纬度维度名称。

    参数说明：
    - TerraClimate 常见维度名为 lon 和 lat；
    - 为了兼容其他 NetCDF，也支持 longitude/latitude 或 x/y。
    """
    经度候选 = ["lon", "longitude", "x"]
    纬度候选 = ["lat", "latitude", "y"]

    经度维度 = next((维度 for 维度 in 经度候选 if 维度 in 数据数组.dims or 维度 in 数据数组.coords), None)
    纬度维度 = next((维度 for 维度 in 纬度候选 if 维度 in 数据数组.dims or 维度 in 数据数组.coords), None)

    if 经度维度 is None or 纬度维度 is None:
        raise KeyError(f"无法识别经纬度维度；当前维度为：{list(数据数组.dims)}")

    return 经度维度, 纬度维度


def 识别时间维度(数据数组):
    """
    识别时间维度名称。

    参数说明：
    - TerraClimate 年度文件通常包含 12 个 time，对应 1-12 月；
    - 逐月绘图和统计都依赖该时间维度。
    """
    时间候选 = ["time", "valid_time", "datetime", "date"]
    时间维度 = next((维度 for 维度 in 时间候选 if 维度 in 数据数组.dims), None)

    if 时间维度 is None:
        raise KeyError(f"无法识别时间维度；当前维度为：{list(数据数组.dims)}")

    return 时间维度


def 获取年份和月份(单月PET, 时间维度, 时间索引):
    """
    从 time 坐标中获取年份和月份。

    参数说明：
    - 单月PET：已经按 time 选出的单月 DataArray；
    - 时间维度：时间维度名称，通常为 time；
    - 时间索引：当前是第几个时间片，用作 time 坐标缺失时的兜底月份。
    """
    if 时间维度 in 单月PET.coords:
        时间值 = pd.to_datetime(单月PET[时间维度].values)
        return int(时间值.year), int(时间值.month)

    return 2025, 时间索引 + 1


def 整理TerraClimatePET(PET原始数据):
    """
    整理 TerraClimate PET 数据。

    处理步骤：
    - TerraClimate pet 已经是月尺度 mm 单位，不做 -pev * 1000；
    - 将变量统一命名为 PET_mm，便于输出文件和统计表理解；
    - 如果 PET最小值限制为0 = True，则把异常负值修正为 0。
    """
    PET数据 = PET原始数据.copy()

    if PET最小值限制为0:
        PET数据 = PET数据.where(PET数据 >= 0, 0)

    PET数据.name = "PET_mm"
    PET数据.attrs["long_name"] = "TerraClimate月PET"
    PET数据.attrs["units"] = PET单位
    PET数据.attrs["说明"] = "TerraClimate pet 原始单位已为 mm，本脚本不做符号和单位换算。"
    return PET数据


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


def 计算月统计(单月PET):
    """
    计算福建省范围内单月 PET 的空间统计值。

    统计项说明：
    - 平均值：福建省范围内所有有效栅格的 PET 平均值；
    - 最大值：有效栅格中的最大 PET；
    - 最小值：有效栅格中的最小 PET；
    - 空间累计值：有效栅格 PET 的直接求和；
    - 有效栅格数：参与统计的非空栅格数量。

    注意：
    - 空间累计值是栅格值求和，不是面积加权水量体积；
    - 如果后续需要严格体积统计，应增加栅格面积权重。
    """
    有效数据 = 单月PET.where(单月PET.notnull())
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


def 保存统计表(统计记录列表, 输出目录):
    """
    保存所有月份的 PET 统计表。

    参数说明：
    - 统计记录列表：每个月处理后返回的一行统计结果；
    - 输出目录：fujian_PET_TerraClimate 目录。

    输出说明：
    - 文件名使用中文：福建省TerraClimate PET月统计表.csv；
    - 编码使用 utf-8-sig，便于 Excel 直接打开中文不乱码。
    - 该表汇总所有输入文件、所有月份的统计结果，不再输出图片相关字段。
    """
    统计表路径 = 输出目录 / "福建省TerraClimate PET月统计表.csv"
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
    ]
    统计表 = 统计表[列顺序]
    统计表.to_csv(统计表路径, index=False, encoding="utf-8-sig")
    return 统计表路径


def 处理TerraClimate文件(PET文件路径, 福建省边界, 输出目录):
    """
    裁剪单个 TerraClimate PET 年度文件，并生成该文件对应的图和统计记录。

    参数说明：
    - PET文件路径：当前正在处理的 TerraClimate_pet_YYYY.nc 文件；
    - 福建省边界：已经读取并统一到 EPSG:4326 的边界 GeoDataFrame；
    - 输出目录：fujian_PET_TerraClimate 目录；

    输出说明：
    - 每个年度文件输出一个英文命名的裁剪 NetCDF；
    - 不再输出图片，只统计每个月的数值结果；
    - 返回该年度 12 个月的统计记录列表，最后由主流程汇总成一个 CSV。
    """
    print(f"\n正在处理 TerraClimate PET 文件：{PET文件路径.name}")

    输出nc路径 = 输出目录 / f"{裁剪后NetCDF文件名前缀}{PET文件路径.name}"
    统计记录列表 = []

    with xr.open_dataset(PET文件路径) as 数据集:
        if PET变量名 not in 数据集.data_vars:
            raise KeyError(f"未找到 PET 变量：{PET变量名}；可用变量为：{list(数据集.data_vars)}")

        PET原始数据 = 数据集[PET变量名]
        经度维度, 纬度维度 = 识别空间维度(PET原始数据)
        时间维度 = 识别时间维度(PET原始数据)

        print(f"识别变量：{PET变量名}")
        print(f"识别维度：时间={时间维度}，经度={经度维度}，纬度={纬度维度}")
        print(f"原始数据维度：{dict(PET原始数据.sizes)}")

        # 步骤 1：整理变量并写入空间参考。
        # TerraClimate PET 已经是月尺度 mm 单位，因此这里只做变量规范化和空间元数据设置。
        PET数据 = 整理TerraClimatePET(PET原始数据)
        PET数据 = PET数据.rio.set_spatial_dims(x_dim=经度维度, y_dim=纬度维度)
        PET数据 = PET数据.rio.write_crs("EPSG:4326")

        # 步骤 2：按福建省行政边界裁剪。
        # drop=True 表示只保留福建省边界外包矩形范围内的数据；
        # all_touched=False 表示按默认规则判断栅格是否落入边界。
        PET裁剪后 = PET数据.rio.clip(
            福建省边界.geometry,
            福建省边界.crs,
            drop=True,
            all_touched=False,
        )
        print(f"裁剪后数据维度：{dict(PET裁剪后.sizes)}")

        # 步骤 3：保存裁剪后的年度 NetCDF。
        # 输出文件使用英文命名，避免 netCDF4/HDF5 后端在中文 .nc 文件名上报错。
        输出数据集 = xr.Dataset({"PET_mm": PET裁剪后})
        输出数据集.attrs["数据说明"] = "福建省范围内 TerraClimate PET 裁剪结果。"
        输出数据集.attrs["单位处理"] = "TerraClimate pet 原始单位已为 mm，未进行单位换算。"
        输出数据集.attrs["裁剪范围"] = "福建省行政边界"
        输出数据集.attrs["对应原始文件"] = PET文件路径.name

        编码参数 = 生成NetCDF编码参数(输出数据集)
        print(f"保存裁剪后的年度 NetCDF 文件：{输出nc路径.name}")
        输出数据集.to_netcdf(输出nc路径, encoding=编码参数)

        # 步骤 4：逐月统计。
        # TerraClimate 的每个 time 已经代表一个月，因此这里按时间片逐月计算统计值。
        月份总数 = int(PET裁剪后.sizes[时间维度])
        for 时间索引 in range(月份总数):
            打印进度条("当前文件统计进度", 时间索引, 月份总数)
            单月PET = PET裁剪后.isel({时间维度: 时间索引})
            年份, 月份 = 获取年份和月份(单月PET, 时间维度, 时间索引)

            print(f"统计：{年份}年{月份:02d}月")

            月统计 = 计算月统计(单月PET)
            月统计.update(
                {
                    "年份": 年份,
                    "月份": 月份,
                    "原始文件": PET文件路径.name,
                    "裁剪结果文件": 输出nc路径.name,
                }
            )
            统计记录列表.append(月统计)
            打印进度条("当前文件统计进度", 时间索引 + 1, 月份总数)

    return 统计记录列表


def 主函数():
    """执行福建省 TerraClimate PET 单文件裁剪和统计表生成流程。"""
    打印分隔线()
    print("开始执行：福建省 TerraClimate PET 单文件裁剪和统计表生成")
    print(f"TerraClimate PET 原始文件：{PET原始文件路径}")
    print(f"福建省边界文件：{福建省边界文件}")

    总步骤数 = 5

    # 关键步骤 1：检查输入路径并创建输出目录。
    # 这样可以在正式处理前尽早发现路径问题，避免裁剪到中途才失败。
    打印进度条("关键步骤", 1, 总步骤数)
    输出目录 = 校验输入路径并创建输出目录()
    print(f"输出目录：{输出目录}")

    # 关键步骤 2：读取福建省边界。
    # 边界只需读取一次，后续所有年度文件裁剪都复用该对象。
    打印进度条("关键步骤", 2, 总步骤数)
    福建省边界 = 读取福建省边界()
    print(f"福建省边界要素数量：{len(福建省边界)}")

    # 关键步骤 3：处理写死配置的单个 TerraClimate PET 年度文件。
    # 如果后续要换年份，只需要修改脚本顶部的 PET原始文件路径。
    打印进度条("关键步骤", 3, 总步骤数)
    统计记录列表 = 处理TerraClimate文件(PET原始文件路径, 福建省边界, 输出目录)

    # 关键步骤 4：保存中文统计表。
    # 统计表汇总该年度所有月份，便于直接查看福建省范围内 TerraClimate PET 的月尺度变化。
    打印进度条("关键步骤", 4, 总步骤数)
    统计表路径 = 保存统计表(统计记录列表, 输出目录)

    # 关键步骤 5：输出完成信息。
    打印进度条("关键步骤", 5, 总步骤数)
    打印分隔线()
    print("全部处理完成。")
    print(f"输出目录：{输出目录}")
    print(f"统计表：{统计表路径}")
    print("已输出：英文命名的裁剪 NetCDF 文件。")
    print("该年度所有月份的统计结果已汇总到一个中文 CSV 表格。")


if __name__ == "__main__":
    try:
        主函数()
    except Exception as 错误:
        打印分隔线()
        print("脚本执行失败，请根据以下错误检查路径、变量名、坐标系或 NetCDF 文件结构。")
        print(f"{type(错误).__name__}: {错误}")
        sys.exit(1)
