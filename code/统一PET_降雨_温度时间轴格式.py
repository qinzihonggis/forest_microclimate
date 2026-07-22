import pandas as pd

# 构建统一的月时间轴（1981-01 至 2025-02，你的研究结束时间）
time_index = pd.date_range("2025-01-01", "2025-12-01", freq="MS")

chirps = xr.open_dataset("chirps_monthly_1km.nc")["precip"]
pet    = xr.open_dataset("terraclimate_pet_fujian_1km.nc")["pet"]

# 重新对齐时间轴
chirps = chirps.sel(time=chirps.time.dt.floor("D"))  # 去掉时区等干扰
pet    = pet.sel(time=pet.time.dt.floor("D"))

# 找到共同时间范围
common_start = max(chirps.time.values[0], pet.time.values[0])
common_end   = min(chirps.time.values[-1], pet.time.values[-1])

chirps = chirps.sel(time=slice(common_start, common_end))
pet    = pet.sel(time=slice(common_start, common_end))

print("统一后时间范围：", common_start, "至", common_end)
print("时间步数：", len(chirps.time))