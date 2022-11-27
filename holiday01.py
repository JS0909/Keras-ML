import pandas as pd
import datetime
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import AbstractHolidayCalendar, nearest_workday, Holiday 
from pandas.tseries.holiday import USMartinLutherKingJr, USPresidentsDay, USMemorialDay 
from pandas.tseries.holiday import USLaborDay, USColumbusDay, USThanksgivingDay

# usb = CustomBusinessDay(calendar = USFederalHolidayCalendar())
# start_end_date_range = pd.date_range('7/1/2018', '7/10/2018', freq=usb)
# print(start_end_date_range)
# # Output:
# # DatetimeIndex(['2018-07-02', '2018-07-03', '2018-07-05', '2018-07-06', '2018-07-09', '2018-07-10'],               dtype='datetime64[ns]', freq='C')

# class USFederalHolidayCalendar(AbstractHolidayCalendar):
#     """
#     US Federal Government Holiday Calendar based on rules specified by:
#     https://www.opm.gov/policy-data-oversight/
#        snow-dismissal-procedures/federal-holidays/
#     """
#     rules = [
#         Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
#         USMartinLutherKingJr,
#         USPresidentsDay,
#         USMemorialDay,
#         Holiday("July 4th", month=7, day=4, observance=nearest_workday),
#         USLaborDay,
#         USColumbusDay,
#         Holiday("Veterans Day", month=11, day=11, observance=nearest_workday),
#         USThanksgivingDay,
#         Holiday("Christmas", month=12, day=25, observance=nearest_workday),
#     ]

class KoreanHoliday(AbstractHolidayCalendar):
    rules = [
        Holiday('양력설', month=1, day=1),
        Holiday('삼일절', month=3, day=1),
        Holiday('노동절', month=5, day=1),
        Holiday('어린이', month=5, day=5),
        Holiday('현충일', month=6, day=6),
        Holiday('광복절', month=8, day=15),
        Holiday('개천절', month=10, day=3),
        Holiday('한글날', month=10, day=9),
        Holiday('성탄절', month=12, day=25),
        # 2021년 9월부터 observance = nearest_workday       // if문으로 구성하자.

        # 2021년부터 식목일 휴일에서 빠짐.
        Holiday("2021 설날1", year=2021, month=2, day=11),
        Holiday("2021 설날2", year=2021, month=2, day=12), 
        Holiday("2021 설날3", year=2021, month=2, day=13), 
        Holiday("2021 부처님오신날", year=2021, month=5, day=19), 
        Holiday("2021 광복절 대체", year=2021, month=8, day=16),  
        Holiday("2021 추석1", year=2021, month=9, day=20),  
        Holiday("2021 추석2", year=2021, month=9, day=21),   
        Holiday("2021 추석3", year=2021, month=9, day=22),  
        Holiday("2021 개천절 대체", year=2021, month=10, day=4),
        Holiday("2021 한글날 대체", year=2021, month=10, day=11),   
        # 휴장일 아직 안너놈. 짐 증권쓸거 아니고 지하철유동 쓸거라 휴장일들 주석처리해놈.
        
        Holiday("2020 설날1", year=2020, month=1, day=24),
        Holiday("2020 설날2", year=2020, month=1, day=25), 
        Holiday("2020 설날3", year=2020, month=1, day=26), 
        Holiday("2020 설날 대체", year=2020, month=1, day=27), 
        Holiday("2020 국회의원 선거", year=2020, month=4, day=15), 
        Holiday("2020 부처님오신날", year=2020, month=4, day=30), 
        Holiday("2020 광복절 대체", year=2020, month=8, day=17),   
        Holiday("2020 추석1", year=2020, month=9, day=30),  
        Holiday("2020 추석2", year=2020, month=10, day=1),   
        Holiday("2020 추석3", year=2020, month=10, day=2),  
        # Holiday("2020 휴장일", year=2020, month=12, day=31),  

        Holiday("2019 설날1", year=2019, month=2, day=4),  
        Holiday("2019 설날2", year=2019, month=2, day=5),  
        Holiday("2019 설날3", year=2019, month=2, day=6),  
        Holiday("2019 부처님오신날", year=2019, month=5, day=6),  
        Holiday("2019 추석1", year=2019, month=9, day=12),  
        Holiday("2019 추석2", year=2019, month=9, day=13),  
        Holiday("2019 추석3", year=2019, month=9, day=14),  
        # Holiday("2019 휴장일", year=2019, month=12, day=31),  

        # 2018년 대체공휴일 도입 : 설날, 추석, 어린이날
        Holiday("2018 설날1", year=2018, month=2, day=15),  
        Holiday("2018 설날2", year=2018, month=2, day=16),  
        Holiday("2018 설날3", year=2018, month=2, day=17),  
        Holiday("2018 부처님오신날", year=2018, month=5, day=22),  
        Holiday("2018 지방 선거", year=2018, month=6, day=13),  
        Holiday("2018 추석1", year=2018, month=9, day=23),  
        Holiday("2018 추석2", year=2018, month=9, day=24),  
        Holiday("2018 추석3", year=2018, month=9, day=25),  
        Holiday("2018 추석4", year=2018, month=9, day=26),  
        # Holiday("2018 휴장일", year=2018, month=12, day=31),  

        Holiday("2017 설날1", year=2017, month=1, day=27),  
        Holiday("2017 설날2", year=2017, month=1, day=28),  
        Holiday("2017 설날3", year=2017, month=1, day=29),  
        Holiday("2017 부처님오신날", year=2017, month=5, day=3),  
        Holiday("2017 추석1", year=2017, month=10, day=3),  
        Holiday("2017 추석2", year=2017, month=10, day=4),  
        Holiday("2017 추석3", year=2017, month=10, day=5),  
        # Holiday("2017 휴장일", year=2017, month=12, day=29),  

        # 제헌절 휴일은 2016년까지????
        Holiday("2016 설날1", year=2016, month=2, day=7),
        Holiday("2016 설날2", year=2016, month=2, day=8), 
        Holiday("2016 설날3", year=2016, month=2, day=9),
        Holiday("2016 설날4", year=2016, month=2, day=10),  # 설날 대체휴무  
        Holiday("2016 국회의원 선거", year=2016, month=4, day=13), 
        Holiday("2016 부처님오신날", year=2016, month=5, day=14), 
        Holiday("2016 제헌절", year=2016, month=7, day=17), # 휴무여부 확인해야함 / 어짜피 일요일
        Holiday("2016 추석1", year=2016, month=9, day=14), 
        Holiday("2016 추석2", year=2016, month=9, day=15), 
        Holiday("2016 추석3", year=2016, month=9, day=16), 
        # Holiday("2016 휴장일", year=2016, month=12, day=30), 
    ]

myday = CustomBusinessDay(calendar=KoreanHoliday())
# start_end_date_range = pd.date_range('9/11/2021', periods=19, freq=myday) # freq : 휴일뺀
# start_end_date_range = pd.date_range('1/1/2016', end='9/30/2021', freq=myday)   # 공휴일 뺀
start_end_date_range = pd.date_range('1/1/2016', end='9/30/2021')   # 휴일포함 전체
print(start_end_date_range)


print("=====================================")
print(start_end_date_range[0], '//', start_end_date_range[-1], '//', start_end_date_range[-1] - start_end_date_range[0])
print("=====================================")

# chu_seok = ['2021-09-20', '2021-09-21', '2021-09-22']
# myday2 = CustomBusinessDay(holiday=chu_seok)
# print(myday2)

start_date = start_end_date_range[0]
end_date = start_end_date_range[-1]

print(start_date)       # 2016-01-04 00:00:00
print(end_date)         # 2021-09-30 00:00:00

# start_date = datetime.datetime.strftime(start_end_date_range[0], '%d%b%Y')
# end_date = datetime.datetime.strftime(start_end_date_range[-1], '%d%b%Y')    # str

# print(end_date)
# print(type(end_date))

# start_date = datetime.datetime.strptime(start_date, '%d%b%Y')
# end_date = datetime.datetime.strptime(end_date, '%d%b%Y')

# print(end_date)
# print(type(end_date))       # <class 'datetime.datetime'>

calendar = KoreanHoliday()
holidays = calendar.holidays(start_date, end_date)
print(holidays)         # DatetimeIndex([], dtype='datetime64[ns]', freq=None)

holiday_date_list = holidays.date.tolist()
# print(holiday_date_list)

xxx = np.busday_count(start_date.date(), end_date.date())
print("모든 영업일수(토일뺀) : ", xxx)      # 1498
zzz = np.busday_count(start_date.date(), end_date.date(), holidays=holiday_date_list)
print("휴일까지 뺀 영업일수 : ", zzz)      # 1422

# yyy = np.is_busday(start_date.date(), end_date.date(), holidays=holiday_date_list)
# print(yyy)

# 
# print(start_date.date(), end_date.date())       # 2016-01-04 2021-09-30

# print(start_date)           # 2016-01-04 00:00:00     
# print(type(start_date))     # <class 'pandas._libs.tslibs.timestamps.Timestamp'>


# zzz = np.busday_count(start_date.date(), start_date.date(), holidays=holiday_date_list)
# print(zzz)
