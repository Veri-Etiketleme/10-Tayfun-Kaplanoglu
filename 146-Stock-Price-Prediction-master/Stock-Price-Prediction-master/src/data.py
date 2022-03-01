# first install the required package via this cli command:
# pip install requests

import intrinio
intrinio.client.username = "65581269ecb14b6c8fdeb2db8a8e9a72"
intrinio.client.password = "dd0b3291ad3fd7ff5ccd5f07c141a75c"

company = "Apple"
ticker = "AAPL"
start_date =  "2018-07-01"

prices = intrinio.prices(ticker, start_date)
info = intrinio.companies(ticker)
search = intrinio.companies(query=company)
financials = intrinio.financials(ticker)

print(info)
print(search)
print(financials)