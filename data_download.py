import datetime
from datetime import timedelta
import time
import gdax
import csv


def format_date(dt):
    return dt.isoformat() + '.000000Z'


public_client = gdax.PublicClient()

# res = public_client.get_product_historic_rates('BTC-USD', granularity=60,)
#
# print(res)

temp = datetime.date(year=2016, month=1, day=1)
fromts = time.mktime(temp.timetuple())

start = datetime.datetime.fromtimestamp(fromts)
now = datetime.datetime.now()

pid = 'BTC-USD'

with open('test.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    end = start
    while start < now:
        print(format_date(start))

        end = start + timedelta(minutes=100)
        res = public_client.get_product_historic_rates(product_id=pid, start=format_date(start), end=format_date(end),
                                                       granularity=60)
        for linevalue in res:
            csvwriter.writerow((linevalue[0], linevalue[4], linevalue[5]))
        time.sleep(0.3)
        start = start + timedelta(minutes=100)

# 2014-11-06T10:34:47.123456Z
# 2018-05-01T00:00:00.000000Z
