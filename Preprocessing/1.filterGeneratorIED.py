import pandas as pd
import datetime
from dateutil import parser

separator = ','
df = pd.read_csv("Modbus_Dataset/attack/compromised-ied/attack logs/allAttacks.csv", sep=separator,
                 header=0, index_col=None)

end = False
command = ""
for i in df.index:
    date_from = df['Timestamp'][i].replace(' ', 'T').split('.')[0]

    if len(df.index) - (i + 1) == 0:
        date_temp = parser.parse(df['Timestamp'][i].replace(' ', 'T'))
        time_change = datetime.timedelta(minutes=60)
        date_to = (date_temp + time_change).isoformat().split('.')[0]
        end = True
    elif len(df.index) - (i + 1) == 1:
        date_to = df['Timestamp'][i + 1].replace(' ', 'T').split('.')[0]
    else:
        for j in range(1, len(df.index)-(i+1)):
            if df['TransactionID'][i+j] != df['TransactionID'][i]:
                date_to = df['Timestamp'][i+j].replace(' ', 'T').split('.')[0]
                break

    command += ('(mbtcp.trans_id == {} && frame.time >= "{}Z" && frame.time < "{}Z")'
                .format(df['TransactionID'][i], date_from, date_to))

    if not end:
        command += ' || '

file = open("Modbus_Dataset/attack/compromised-ied/attack logs/command.txt", "w")
file.write(command)
