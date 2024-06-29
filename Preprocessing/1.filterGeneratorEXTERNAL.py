import dateutil
import pandas as pd
import datetime
from dateutil import parser

separator = ','
df = pd.read_csv("Modbus_Dataset/attack/external/attack logs/02-01-2023-1.csv", sep=separator,
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
        file_temp = open("Modbus_Dataset/attack/external/attack logs/commands/missingThings.txt", "a")
        file_temp.write("Missing attack type: {}\n".format(attack))
        file_temp.close()


command_fs = "("
command_qf = "("
command_lm = "("
command_reco = "("
command_bf = "("
command_repl = "("
command_pi = "("

current_command = ""
streak_attack = ""
streak_date = ""
streak_date_max = ""
for i in df.index:
    if "Complete" in df['Attack'][i]:
        continue

    date_current = df['Timestamp'][i].replace(' ', 'T').split('.')[0]
    if len(date_current) != 19:  # Alcune date sono formattate male
        continue

    attack_current = attack_recon(df['Attack'][i])

    if streak_attack == "":
        streak_attack = attack_current
        streak_date = date_current
        try:
            date_temp = parser.parse(df['Timestamp'][i].replace(' ', 'T'))
            time_change = datetime.timedelta(seconds=10)
            streak_date_max = (date_temp + time_change).isoformat().split('.')[0]
        except dateutil.parser.ParserError:  # Alcune date sono formattate male
            continue

    elif len(df.index) - (i + 1) == 1:
        current_command = ('(modbus && frame.time >= "{}Z" && frame.time < "{}Z")'
                           .format(streak_date, streak_date_max))

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
        date_changed = (parsed_date_current > parser.parse(streak_date_max))

        if streak_attack == "Brute force":
            if date_changed:
                try:
                    date_temp = parser.parse(streak_date_max)
                    time_change = datetime.timedelta(seconds=10)
                    date_temp = (date_temp + time_change).isoformat().split('.')[0]
                except dateutil.parser.ParserError:  # Alcune date sono formattate male
                    continue

                if parsed_date_current > parser.parse(date_temp):
                    current_command = ('(modbus && frame.time >= "{}Z" && frame.time < "{}Z")'
                                       .format(streak_date, streak_date_max))
                    command_bf += current_command
                    command_bf += ' || '

                    streak_date = date_current
                    try:
                        date_temp = parser.parse(df['Timestamp'][i].replace(' ', 'T'))
                        time_change = datetime.timedelta(seconds=10)
                        streak_date_max = (date_temp + time_change).isoformat().split('.')[0]
                    except dateutil.parser.ParserError:  # Alcune date sono formattate male
                        continue

                else:
                    streak_date_max = date_temp

        else:
            attack_changed = not (attack_current == streak_attack)

            if date_changed or attack_changed:
                current_command = ('(modbus && frame.time >= "{}Z" && frame.time < "{}Z")'
                                   .format(streak_date, streak_date_max))

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

                if date_changed:
                    streak_date = date_current
                    try:
                        date_temp = parser.parse(df['Timestamp'][i].replace(' ', 'T'))
                        time_change = datetime.timedelta(seconds=10)
                        streak_date_max = (date_temp + time_change).isoformat().split('.')[0]
                    except dateutil.parser.ParserError:  # Alcune date sono formattate male
                        continue
                if attack_changed:
                    streak_attack = attack_current

file = open("Modbus_Dataset/attack/external/attack logs/commands/frameStackingCommand.txt", "w")
command_fs += ') && (ip.src == 185.175.0.7 || ip.dst == 185.175.0.7)'
file.write(command_fs)
file.close()
file = open("Modbus_Dataset/attack/external/attack logs/commands/queryFloodingCommand.txt", "w")
command_qf += ') && (ip.src == 185.175.0.7 || ip.dst == 185.175.0.7)'
file.write(command_qf)
file.close()
file = open("Modbus_Dataset/attack/external/attack logs/commands/lengthManipulationCommand.txt", "w")
command_lm += ') && (ip.src == 185.175.0.7 || ip.dst == 185.175.0.7)'
file.write(command_lm)
file.close()
file = open("Modbus_Dataset/attack/external/attack logs/commands/reconnaissanceCommand.txt", "w")
command_reco += ') && (ip.src == 185.175.0.7 || ip.dst == 185.175.0.7)'
file.write(command_reco)
file.close()
file = open("Modbus_Dataset/attack/external/attack logs/commands/bruteForceCommand.txt", "w")
command_bf += ') && (ip.src == 185.175.0.7 || ip.dst == 185.175.0.7)'
file.write(command_bf)
file.close()
file = open("Modbus_Dataset/attack/external/attack logs/commands/baselineReplayCommand.txt", "w")
command_repl += ') && (ip.src == 185.175.0.7 || ip.dst == 185.175.0.7)'
file.write(command_repl)
file.close()
file = open("Modbus_Dataset/attack/external/attack logs/commands/payloadInjectionCommand.txt", "w")
command_pi += ') && (ip.src == 185.175.0.7 || ip.dst == 185.175.0.7)'
file.write(command_pi)
file.close()
