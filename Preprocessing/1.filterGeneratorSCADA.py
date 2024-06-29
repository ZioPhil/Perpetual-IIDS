import dateutil
import pandas as pd
import datetime
from dateutil import parser

CURRENT_DEVICE = "IED4C"

separator = ','
df = pd.read_csv("Modbus_Dataset/attack/compromised-scada/attack logs/{}.csv".format(CURRENT_DEVICE), sep=separator,
                 header=0, index_col=None)


def attack_recon(attack):
    if "Frame Stacking" in attack or "Stacked Modbus Frames" in attack:
        return "Frame Stacking"
    elif "Query flooding" in attack or "Query Flooding" in attack:
        return "Query flooding"
    elif "Length manipulation" in attack:
        return "Length manipulation"
    elif "Recon" in attack:
        return "Recon"
    elif "Brute force" in attack:
        return "Brute force"
    elif "Baseline Replay" in attack or "Replay" in attack:
        return "Baseline Replay"
    elif "Payload Injection" in attack or "Payload injection" in attack:
        return "Payload Injection"
    else:
        file_temp = open("Modbus_Dataset/attack/compromised-scada/attack logs/commands/missingThings.txt", "a")
        file_temp.write("Missing attack type: {}\n".format(attack))
        file_temp.close()


command_fs = ""
command_qf = ""
command_lm = ""
command_reco = ""
command_bf = ""
command_repl = ""
command_pi = ""

current_command = ""
streak_id = -1
streak_attack = ""
streak_date = ""
streak_date_max = ""
invalid_streak = False
for i in df.index:
    if "Complete" in df['Attack'][i]:
        continue

    date_current = df['Timestamp'][i].replace(' ', 'T').split('.')[0]
    if len(date_current) != 19:  # Alcune date sono formattate male
        continue

    attack_current = attack_recon(df['Attack'][i])
    id_current = df['TransactionID'][i]

    if streak_attack == "":
        streak_attack = attack_current
        streak_id = id_current
        streak_date = date_current
        try:
            date_temp = parser.parse(df['Timestamp'][i].replace(' ', 'T'))
            time_change = datetime.timedelta(seconds=60)
            streak_date_max = (date_temp + time_change).isoformat().split('.')[0]
        except dateutil.parser.ParserError:  # Alcune date sono formattate male
            continue

    elif len(df.index) - (i + 1) == 1 and not invalid_streak:
        current_command = ('(mbtcp.trans_id == {} && frame.time >= "{}Z" && frame.time < "{}Z")'
                           .format(streak_id, streak_date, streak_date_max))

        if "Frame Stacking" in streak_attack or "Stacked Modbus Frames" in streak_attack:
            command_fs += current_command
        elif "Query flooding" in streak_attack or "Query Flooding" in streak_attack:
            command_qf += current_command
        elif "Length manipulation" in streak_attack:
            command_lm += current_command
        elif "Recon" in streak_attack:
            command_reco += current_command
        elif "Brute force" in streak_attack:
            command_bf += current_command
        elif "Baseline Replay" in streak_attack or "Replay" in streak_attack:
            command_repl += current_command
        elif "Payload Injection" in streak_attack or "Payload injection" in streak_attack:
            command_pi += current_command

    else:
        parsed_date_current = parser.parse(df['Timestamp'][i].replace(' ', 'T'))
        id_changed = not (id_current == streak_id)
        date_changed = (parsed_date_current > parser.parse(streak_date_max))
        attack_changed = not (attack_current == streak_attack)

        if (id_changed or date_changed) and not invalid_streak:
            current_command = ('(mbtcp.trans_id == {} && frame.time >= "{}Z" && frame.time < "{}Z")'
                               .format(streak_id, streak_date, streak_date_max))

            if "Frame Stacking" in streak_attack or "Stacked Modbus Frames" in streak_attack:
                command_fs += current_command
                command_fs += ' || '
            elif "Query flooding" in streak_attack or "Query Flooding" in streak_attack:
                command_qf += current_command
                command_qf += ' || '
            elif "Length manipulation" in streak_attack:
                command_lm += current_command
                command_lm += ' || '
            elif "Recon" in streak_attack:
                command_reco += current_command
                command_reco += ' || '
            elif "Brute force" in streak_attack:
                command_bf += current_command
                command_bf += ' || '
            elif "Baseline Replay" in streak_attack or "Replay" in streak_attack:
                command_repl += current_command
                command_repl += ' || '
            elif "Payload Injection" in streak_attack or "Payload injection" in streak_attack:
                command_pi += current_command
                command_pi += ' || '

            if id_changed:
                streak_id = id_current
            if date_changed:
                streak_date = date_current
                try:
                    date_temp = parser.parse(df['Timestamp'][i].replace(' ', 'T'))
                    time_change = datetime.timedelta(seconds=60)
                    streak_date_max = (date_temp + time_change).isoformat().split('.')[0]
                except dateutil.parser.ParserError:  # Alcune date sono formattate male
                    continue
            if attack_changed:
                streak_attack = attack_current

        elif (id_changed or date_changed) and invalid_streak:
            streak_attack = attack_current
            streak_id = id_current
            streak_date = date_current
            try:
                date_temp = parser.parse(df['Timestamp'][i].replace(' ', 'T'))
                time_change = datetime.timedelta(seconds=60)
                streak_date_max = (date_temp + time_change).isoformat().split('.')[0]
            except dateutil.parser.ParserError:  # Alcune date sono formattate male
                continue
            invalid_streak = False

        elif attack_changed and not id_changed and not date_changed and not invalid_streak:
            invalid_streak = True

file = open("Modbus_Dataset/attack/compromised-scada/attack logs/commands/{}/frameStackingCommand.txt".format(CURRENT_DEVICE), "w")
file.write(command_fs)
file.close()
file = open("Modbus_Dataset/attack/compromised-scada/attack logs/commands/{}/queryFloodingCommand.txt".format(CURRENT_DEVICE), "w")
file.write(command_qf)
file.close()
file = open("Modbus_Dataset/attack/compromised-scada/attack logs/commands/{}/lengthManipulationCommand.txt".format(CURRENT_DEVICE), "w")
file.write(command_lm)
file.close()
file = open("Modbus_Dataset/attack/compromised-scada/attack logs/commands/{}/reconnaissanceCommand.txt".format(CURRENT_DEVICE), "w")
file.write(command_reco)
file.close()
file = open("Modbus_Dataset/attack/compromised-scada/attack logs/commands/{}/bruteForceCommand.txt".format(CURRENT_DEVICE), "w")
file.write(command_bf)
file.close()
file = open("Modbus_Dataset/attack/compromised-scada/attack logs/commands/{}/baselineReplayCommand.txt".format(CURRENT_DEVICE), "w")
file.write(command_repl)
file.close()
file = open("Modbus_Dataset/attack/compromised-scada/attack logs/commands/{}/payloadInjectionCommand.txt".format(CURRENT_DEVICE), "w")
file.write(command_pi)
file.close()
