import holiday01 as td2

import pandas as pd
import datetime
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import AbstractHolidayCalendar, nearest_workday, Holiday 
from pandas.tseries.holiday import USMartinLutherKingJr, USPresidentsDay, USMemorialDay 
from pandas.tseries.holiday import USLaborDay, USColumbusDay, USThanksgivingDay


myday = CustomBusinessDay(calendar=td2.KoreanHoliday())
# start_end_date_range = pd.date_range('9/11/2021', periods=19, freq=myday) 
start_end_date_range = pd.date_range('1/1/2016', end='9/30/2021', freq=myday)
print(start_end_date_range)

print("=====================================")
print(start_end_date_range[0], '//', start_end_date_range[-1], '//', start_end_date_range[-1] - start_end_date_range[0])
print("=====================================")

# chu_seok = ['2021-09-20', '2021-09-21', '2021-09-22']
# myday2 = CustomBusinessDay(holiday=chu_seok)
# print(myday2)

start_date = start_end_date_range[0]
end_date = start_end_date_range[-1]

# print(start_date)       # 2016-01-04 00:00:00     # 1월1일 금토일? 토일월?  
# print(end_date)         # 2021-09-30 00:00:00  


start_date = datetime.datetime.strftime(start_end_date_range[0], '%d%b%Y')
end_date = datetime.datetime.strftime(start_end_date_range[-1], '%d%b%Y')    # str

print(start_date)       # 04Jan2016
print(end_date)         # 30Sep2021
print(type(end_date))   # <class 'str'>

"""
# start_date = datetime.datetime.strptime(start_date, '%d%b%Y')
# end_date = datetime.datetime.strptime(end_date, '%d%b%Y')

# print(end_date)
# print(type(end_date))       # <class 'datetime.datetime'>

calendar = td2.KoreanHoliday()
holidays = calendar.holidays(start_date, end_date)
# print(holidays)         # DatetimeIndex([], dtype='datetime64[ns]', freq=None)

holiday_date_list = holidays.date.tolist()
# print(holiday_date_list)
xxx = np.busday_count(start_date.date(), end_date.date())
print('토일을 뺀 수 : ', xxx)      # 13
zzz = np.busday_count(start_date.date(), end_date.date(), holidays=holiday_date_list)
print('공휴일 뺀 수 : ', zzz)      # 10

print(start_date.date(), end_date.date())       # 2021-09-13 2021-09-30

"""



