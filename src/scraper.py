import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
from datetime import datetime 
from data_scraping import Data_scrapper

start = '20181016'
end = '20190410'

scrapper = Data_scrapper(start, end)

result = scrapper.get_timeframe_data(sleep=10, name='season_2018')

