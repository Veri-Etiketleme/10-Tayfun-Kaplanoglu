

#Sending SMS using fast2sms

import requests
url = "https://www.fast2sms.com/dev/bulk"
payload = "sender_id=FSTSMS&message=Switching off TV in 15 sec&language=english&route=p&numbers=#Mobile_Number"
headers = {
'authorization': "p5wDW4hd3mOt8CR9LGkYFzET1lVKoZNrfqQ0j2cMHJaByxeISssxY2mdeVPKR01oL3hqvM9g7bGarD4J",
'Content-Type': "application/x-www-form-urlencoded",
'Cache-Control': "no-cache",
}
response = requests.request("POST", url, data=payload, headers=headers)
print(response.text)
