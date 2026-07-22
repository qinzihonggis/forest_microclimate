import pandas as pd
import numpy as np
from spei import spei
import scipy.stats as sps
import pyet    # pip install pyet，专门计算PET

# ============================================================
# 第一步：读入降水和气温数据
# ============================================================
precip_df = pd.read_csv("chirps_monthly.csv",  index_col=0, parse_dates=True)
temp_df   = pd.read_csv("era5_temp_monthly.csv", index_col=0, parse_dates=True)
# temp_df 是 ERA5-Land 的月均气温（℃），格点和时间索引需与 precip_df 一致

# ============================================================
# 第二步：计算 PET（Thornthwaite 方法，只需要月均温和纬度）
# ============================================================
# Thornthwaite 是最简单的 PET 方法，只需要月均温
# 如果你有太阳辐射数据，可以改用 Hargreaves 或 Penman-Monteith（更精确）

# 福建省纬度范围大约 23.5°N – 28.3°N
# 每个格点的纬度从 NetCDF 坐标里读取，这里示例用单个格点

lat = 26.5    # 替换为实际格点纬度（°N）

# pyet.thornthwaite 输入：月均温 Series（℃），纬度（°）
# 返回：月 PET（mm/month）
pet_series = pyet.thornthwaite(
    tmean = temp_df["grid_001"],   # 月均温
    lat   = lat
)

# ============================================================
# 第三步：计算水分平衡 D = P - PET
# ============================================================
D = precip_df["grid_001"] - pet_series
# D > 0：水分盈余；D < 0：水分亏缺（干旱）

# ============================================================
# 第四步：计算 SPEI
# ============================================================
# 参数和 SPI 完全一致，timescale 同样可以选 3 / 6 / 12
spei_obj = spei(D, timescale=3)
spei_values = spei_obj.si

print(spei_values.tail(20))

# ============================================================
# 第五步：批量对所有格点计算 SPEI
# ============================================================
# 注意：每个格点的纬度不同，需要从格点信息表读取
grid_info = pd.read_csv("grid_latlon.csv")   # 含 grid_id, lat, lon 三列

spei_all = pd.DataFrame(index=precip_df.index)

for col in precip_df.columns:
    lat_i = grid_info.loc[grid_info["grid_id"] == col, "lat"].values[0]
    
    pet_i = pyet.thornthwaite(tmean=temp_df[col], lat=lat_i)
    D_i   = precip_df[col] - pet_i
    
    if D_i.dropna().__len__() > 24:
        spei_obj = spei(D_i.dropna(), timescale=3)
        spei_all[col] = spei_obj.si
    else:
        spei_all[col] = np.nan

spei_all.to_csv("SPEI3_results.csv")
print("SPEI 计算完成，已保存到 SPEI3_results.csv")