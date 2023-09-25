import pandas as pd
from scipy.interpolate import interp1d

excel_file = pd.ExcelFile('PMA_angle.xlsx')
df_pma_angle = excel_file.parse('Sheet1', usecols="B:C", header=None,nrows=200)

def return_simulation_pma_angle(df_pma_angle,voltage_65535):
    #pma_angle = df_pma_angle[1].interpolate(method='linear', limit_direction='both', limit_area='inside')
    interpolated_function = interp1d(df_pma_angle[1], df_pma_angle[2], kind='linear', fill_value='extrapolate')
    pma_angle = interpolated_function(voltage_65535)
    return pma_angle

print(return_simulation_pma_angle(df_pma_angle,12844))
