import os.path
import pandas as pd
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP
from scapy.contrib.modbus import (ModbusADUResponse, ModbusADURequest,
                                  ModbusPDU01ReadCoilsRequest, ModbusPDU01ReadCoilsResponse,
                                  ModbusPDU02ReadDiscreteInputsRequest, ModbusPDU02ReadDiscreteInputsResponse,
                                  ModbusPDU03ReadHoldingRegistersRequest, ModbusPDU03ReadHoldingRegistersResponse,
                                  ModbusPDU04ReadInputRegistersRequest, ModbusPDU04ReadInputRegistersResponse,
                                  ModbusPDU05WriteSingleCoilRequest, ModbusPDU05WriteSingleCoilResponse,
                                  ModbusPDU06WriteSingleRegisterRequest, ModbusPDU06WriteSingleRegisterResponse,
                                  ModbusPDU01ReadCoilsError, ModbusPDU02ReadDiscreteInputsError,
                                  ModbusPDU03ReadHoldingRegistersError, ModbusPDU04ReadInputRegistersError,
                                  ModbusPDU05WriteSingleCoilError, ModbusPDU06WriteSingleRegisterError)

fmt_modbus = ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},'
              '{22},{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},'
              '{42},{43},{44},{45},{46},{47},{48},{49},{50},{51},{52},{53},{54},{55},{56}')


def render_csv_row(pkt_sc, current_mac, ether_pkt_sc, ip_pkt_sc, tcp_pkt_sc, modbus_pkt_sc, data_type, attack_type,
                   f_modbus):
    pkt_len = len(pkt_sc)
    # print(ether_pkt_sc.show(dump=True))

    timestamp = ether_pkt_sc.time
    src_mac = ether_pkt_sc.src
    dst_mac = ether_pkt_sc.dst

    flow = ""
    if current_mac == src_mac:
        flow = "Sent"
    elif current_mac == dst_mac:
        flow = "Received"

    ip_version = ip_pkt_sc.version
    ip_ihl = ip_pkt_sc.ihl
    ip_tos = ip_pkt_sc.tos
    ip_len = ip_pkt_sc.len
    ip_id = ip_pkt_sc.id
    ip_flags = ip_pkt_sc.flags
    ip_frag = ip_pkt_sc.frag
    ip_ttl = ip_pkt_sc.ttl
    ip_chksum = ip_pkt_sc.chksum
    src_ip = ip_pkt_sc.src
    dst_ip = ip_pkt_sc.dst

    src_port = tcp_pkt_sc.sport
    dst_port = tcp_pkt_sc.dport
    tcp_seq = tcp_pkt_sc.seq
    tcp_ack = tcp_pkt_sc.ack
    tcp_dataofs = tcp_pkt_sc.dataofs
    tcp_reserved = tcp_pkt_sc.reserved
    tcp_flags = tcp_pkt_sc.flags
    tcp_window = tcp_pkt_sc.window
    tcp_chksum = tcp_pkt_sc.chksum
    tcp_urgptr = tcp_pkt_sc.urgptr

    options = tcp_pkt_sc.options
    tcp_option_mss = ''
    tcp_option_sackok = 0
    tcp_option_timestamp = ''
    tcp_option_timestamp_echoreply = ''
    tcp_option_nop_count = 0
    tcp_option_wscale = ''
    tcp_option_sack_left = ''
    tcp_option_sack_right = ''
    for tup in options:
        if tup[0] == 'MSS':
            tcp_option_mss = tup[1]
        elif tup[0] == 'SAckOK':
            tcp_option_sackok += 1
        elif tup[0] == 'Timestamp':
            tcp_option_timestamp = tup[1][0]
            tcp_option_timestamp_echoreply = tup[1][1]
        elif tup[0] == 'NOP':
            tcp_option_nop_count += 1
        elif tup[0] == 'WScale':
            tcp_option_wscale = tup[1]
        elif tup[0] == 'SAck':
            tcp_option_sack_left = tup[1][0]
            tcp_option_sack_right = tup[1][1]
        else:
            file = open("missingThings.txt", "a")
            file.write("Missing TCP option: {}\n".format(tup[0]))
            file.close()

    proto_name = "MODBUS"
    modbus_trans_id = modbus_pkt_sc.transId
    modbus_proto_id = modbus_pkt_sc.protoId
    modbus_len = modbus_pkt_sc.len
    modbus_unit_id = modbus_pkt_sc.unitId

    if (ModbusPDU01ReadCoilsRequest in tcp_pkt_sc or ModbusPDU02ReadDiscreteInputsRequest in tcp_pkt_sc
            or ModbusPDU03ReadHoldingRegistersRequest in tcp_pkt_sc
            or ModbusPDU04ReadInputRegistersRequest in tcp_pkt_sc):
        if ModbusPDU01ReadCoilsRequest in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU01ReadCoilsRequest]
            modbus_pdu = 1
            modbus_type = "Request"
        elif ModbusPDU02ReadDiscreteInputsRequest in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU02ReadDiscreteInputsRequest]
            modbus_pdu = 2
            modbus_type = "Request"
        elif ModbusPDU03ReadHoldingRegistersRequest in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU03ReadHoldingRegistersRequest]
            modbus_pdu = 3
            modbus_type = "Request"
        else:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU04ReadInputRegistersRequest]
            modbus_pdu = 4
            modbus_type = "Request"

        modbus_func_code = modbus_request_pkt_sc.funcCode
        modbus_start_addr = modbus_request_pkt_sc.startAddr
        modbus_quantity = modbus_request_pkt_sc.quantity

        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac, dst_mac, src_ip,
                                dst_ip, src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id,
                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                modbus_unit_id, modbus_func_code, modbus_start_addr, modbus_quantity,
                                "", "", "", "", "", "", "", "", "", "", "", data_type, attack_type),
              file=f_modbus)
        return True

    elif (ModbusPDU05WriteSingleCoilRequest in tcp_pkt_sc
          or ModbusPDU05WriteSingleCoilResponse in tcp_pkt_sc):
        if ModbusPDU05WriteSingleCoilRequest in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU05WriteSingleCoilRequest]
            modbus_pdu = 5
            modbus_type = "Request"
        else:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU05WriteSingleCoilResponse]
            modbus_pdu = 5
            modbus_type = "Response"

        modbus_func_code = modbus_request_pkt_sc.funcCode
        modbus_output_addr = modbus_request_pkt_sc.outputAddr
        modbus_output_value = modbus_request_pkt_sc.outputValue

        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac, dst_mac, src_ip,
                                dst_ip, src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id,
                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                modbus_unit_id, modbus_func_code, "", "", modbus_output_addr,
                                modbus_output_value, "", "", "", "", "", "", "", "", "", data_type, attack_type),
              file=f_modbus)
        return True

    elif ModbusPDU01ReadCoilsResponse in tcp_pkt_sc:
        modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU01ReadCoilsResponse]
        modbus_pdu = 1
        modbus_type = "Response"

        modbus_func_code = modbus_request_pkt_sc.funcCode
        modbus_byte_count = modbus_request_pkt_sc.byteCount

        modbus_coil_status_0 = 0
        modbus_coil_status_1 = 0
        for status in modbus_request_pkt_sc.coilStatus:
            status = '{:0>8}'.format(str(bin(status))[2:])
            for number in status:
                if int(number) == 0:
                    modbus_coil_status_0 += 1
                else:
                    modbus_coil_status_1 += 1

        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac, dst_mac, src_ip,
                                dst_ip, src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id,
                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                modbus_unit_id, modbus_func_code, "", "", "", "", modbus_byte_count,
                                modbus_coil_status_0, modbus_coil_status_1, "", "", "", "", "", "",
                                data_type, attack_type),
              file=f_modbus)
        return True

    elif ModbusPDU02ReadDiscreteInputsResponse in tcp_pkt_sc:
        modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU02ReadDiscreteInputsResponse]
        modbus_pdu = 2
        modbus_type = "Response"

        modbus_func_code = modbus_request_pkt_sc.funcCode
        modbus_byte_count = modbus_request_pkt_sc.byteCount

        modbus_input_status_0 = 0
        modbus_input_status_1 = 0
        for status in modbus_request_pkt_sc.inputStatus:
            status = '{:0>8}'.format(str(bin(status))[2:])
            for number in status:
                if int(number) == 0:
                    modbus_input_status_0 += 1
                else:
                    modbus_input_status_1 += 1

        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac, dst_mac, src_ip,
                                dst_ip, src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id,
                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                modbus_unit_id, modbus_func_code, "", "", "", "", modbus_byte_count, "",
                                "", modbus_input_status_0, modbus_input_status_1, "", "", "", "",
                                data_type, attack_type),
              file=f_modbus)
        return True

    elif (ModbusPDU03ReadHoldingRegistersResponse in tcp_pkt_sc
          or ModbusPDU04ReadInputRegistersResponse in tcp_pkt_sc):
        if ModbusPDU03ReadHoldingRegistersResponse in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU03ReadHoldingRegistersResponse]
            modbus_pdu = 3
            modbus_type = "Response"
        else:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU04ReadInputRegistersResponse]
            modbus_pdu = 4
            modbus_type = "Response"

        modbus_func_code = modbus_request_pkt_sc.funcCode
        modbus_byte_count = modbus_request_pkt_sc.byteCount
        modbus_register_val = 0
        for val in modbus_request_pkt_sc.registerVal:
            if val != 9056:
                modbus_register_val += val

        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac, dst_mac, src_ip,
                                dst_ip, src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id,
                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                modbus_unit_id, modbus_func_code, "", "", "", "", modbus_byte_count, "",
                                "", "", "", modbus_register_val, "", "", "", data_type, attack_type),
              file=f_modbus)
        return True

    elif (ModbusPDU06WriteSingleRegisterRequest in tcp_pkt_sc
          or ModbusPDU06WriteSingleRegisterResponse in tcp_pkt_sc):
        if ModbusPDU06WriteSingleRegisterRequest in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU06WriteSingleRegisterRequest]
            modbus_pdu = 6
            modbus_type = "Request"
        else:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU06WriteSingleRegisterResponse]
            modbus_pdu = 6
            modbus_type = "Response"

        modbus_func_code = modbus_request_pkt_sc.funcCode
        modbus_register_addr = modbus_request_pkt_sc.registerAddr
        modbus_register_value = modbus_request_pkt_sc.registerValue

        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac, dst_mac, src_ip,
                                dst_ip, src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id,
                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                modbus_unit_id, modbus_func_code, "", "", "", "", "", "", "", "",
                                "", "", modbus_register_addr, modbus_register_value, "", data_type, attack_type),
              file=f_modbus)
        return True

    elif (ModbusPDU01ReadCoilsError in tcp_pkt_sc or ModbusPDU02ReadDiscreteInputsError in tcp_pkt_sc
          or ModbusPDU03ReadHoldingRegistersError in tcp_pkt_sc
          or ModbusPDU04ReadInputRegistersError in tcp_pkt_sc
          or ModbusPDU05WriteSingleCoilError in tcp_pkt_sc
          or ModbusPDU06WriteSingleRegisterError in tcp_pkt_sc):
        if ModbusPDU01ReadCoilsError in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU01ReadCoilsError]
            modbus_pdu = 1
            modbus_type = "Error"
        elif ModbusPDU02ReadDiscreteInputsError in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU02ReadDiscreteInputsError]
            modbus_pdu = 2
            modbus_type = "Error"
        elif ModbusPDU03ReadHoldingRegistersError in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU03ReadHoldingRegistersError]
            modbus_pdu = 3
            modbus_type = "Error"
        elif ModbusPDU04ReadInputRegistersError in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU04ReadInputRegistersError]
            modbus_pdu = 4
            modbus_type = "Error"
        elif ModbusPDU05WriteSingleCoilError in tcp_pkt_sc:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU05WriteSingleCoilError]
            modbus_pdu = 5
            modbus_type = "Error"
        else:
            modbus_request_pkt_sc = modbus_pkt_sc[ModbusPDU06WriteSingleRegisterError]
            modbus_pdu = 6
            modbus_type = "Error"

        modbus_func_code = modbus_request_pkt_sc.funcCode
        modbus_except_code = modbus_request_pkt_sc.exceptCode

        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac, dst_mac, src_ip,
                                dst_ip, src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id,
                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                modbus_unit_id, modbus_func_code, "", "", "", "", "", "", "", "", "", "", "", "",
                                modbus_except_code, data_type, attack_type),
              file=f_modbus)
        return True

    else:
        file = open("missingThings.txt", "a")
        file.write("Missing MODBUS protocol:\n {}".format(tcp_pkt_sc.show(dump=True)))
        file.close()
        return False


def format_attack(attack_type):
    if "Baseline Replay" in attack_type or "Replay" in attack_type:
        return "Baseline Replay"
    elif "Payload Injection" in attack_type or "Payload injection" in attack_type:
        return "Payload Injection"
    elif "Delay Response" in attack_type:
        return "Delay Response"
    elif "False Data Injection" in attack_type:
        return "False Data Injection"
    elif "Query flooding" in attack_type or "Query Flooding" in attack_type:
        return "Query Flooding"
    elif "Length manipulation" in attack_type:
        return "Length Manipulation"
    elif "Frame Stacking" in attack_type or "Stacked Modbus Frames" in attack_type:
        return "Frame Stacking"
    elif "Brute force" in attack_type:
        return "Brute Force"
    elif "Recon" in attack_type:
        return "Reconnaissance"
    else:
        file = open("missingThings.txt", "a")
        file.write("Missing attack type: {}\n".format(attack_type))
        file.close()
        return attack_type


def main():
    count = 0
    directories = ["Modbus_Dataset/attack/compromised-ied/trust-scada-hmi/attack",
                   "Modbus_Dataset/attack/compromised-scada/ied1a/attack",
                   "Modbus_Dataset/attack/compromised-scada/ied1b/attack",
                   "Modbus_Dataset/attack/compromised-scada/ied4c/attack",
                   "Modbus_Dataset/attack/external/ied1a/attack"]
    attack_logs = ["Modbus_Dataset/attack/compromised-ied/attack logs/allAttacks.csv",
                   "Modbus_Dataset/attack/compromised-scada/attack logs/IED1A.csv",
                   "Modbus_Dataset/attack/compromised-scada/attack logs/IED1B.csv",
                   "Modbus_Dataset/attack/compromised-scada/attack logs/IED4C.csv",
                   "Modbus_Dataset/attack/external/attack logs/02-01-2023-1.csv"]
    output_files = ["temp/attacksOnHMIfromIED1B.csv",
                    "temp/attacksOnIED1AfromSCADA.csv",
                    "temp/attacksOnIED1BfromSCADA.csv",
                    "temp/attacksOnIED4CfromSCADA.csv",
                    "temp/attacksOnIED1AfromEXT.csv"]
    macs = ["02:42:b9:af:00:02",
            "02:42:b9:af:00:04",
            "02:42:b9:af:00:05",
            "02:42:b9:af:00:08",
            "02:42:b9:af:00:04"]
    outputs = []

    for directory in directories:
        i = directories.index(directory)

        files = [f for f in os.listdir(directory)]
        files.sort()
        df = pd.read_csv(attack_logs[i], sep=',', header=0, index_col=None)
        output_file = open(output_files[i], "w")

        print("proto_name,flow,modbus_pdu,modbus_type,timestamp,pkt_len,src_mac,dst_mac,src_ip,dst_ip,src_port,"
              "dst_port,tcp_chksum,ip_version,ip_ihl,ip_tos,ip_len,ip_id,ip_flags,ip_frag,ip_ttl,ip_chksum,tcp_seq,"
              "tcp_ack,tcp_dataofs,tcp_reserved,tcp_flags,tcp_window,tcp_urgptr,tcp_option_mss,tcp_option_sackok,"
              "tcp_option_timestamp,tcp_option_timestamp_echoreply,tcp_option_nop_count,tcp_option_wscale,"
              "tcp_option_sack_left,tcp_option_sack_right,modbus_trans_id,modbus_proto_id,modbus_len,modbus_unit_id,"
              "modbus_func_code,"
              "modbus_start_addr,modbus_quantity,modbus_output_addr,modbus_output_value,modbus_byte_count,"
              "modbus_coil_status_0,modbus_coil_status_1,modbus_input_status_0,modbus_input_status_1,"
              "modbus_register_val,modbus_register_addr,modbus_register_value,modbus_except_code,"
              "data_type,attack_type", file=output_file)

        csv_i = 0
        for file in files:
            filename = file
            file = os.path.join(os.path.abspath(directory), file)

            frame_num = 0
            ignored_packets = 0
            for (pkt_scapy, _) in RawPcapReader(file):
                try:
                    frame_num += 1

                    ether_pkt_sc = Ether(pkt_scapy)
                    ip_pkt_sc = ether_pkt_sc[IP]
                    tcp_pkt_sc = ip_pkt_sc[TCP]
                    if ModbusADUResponse in tcp_pkt_sc:
                        modbus_pkt_sc = tcp_pkt_sc[ModbusADUResponse]
                    else:
                        modbus_pkt_sc = tcp_pkt_sc[ModbusADURequest]
                    trans_id = modbus_pkt_sc.transId

                    if i == 0:
                        if trans_id == df['TransactionID'][csv_i]:
                            attack_type = format_attack(df['Attack'][csv_i])
                        else:
                            csv_i_old = csv_i
                            while (df['TransactionID'][csv_i] == df['TransactionID'][csv_i_old]
                                   or "Complete" in df['Attack'][csv_i]
                                   or len(df['Timestamp'][csv_i].replace(' ', 'T').split('.')[0]) != 19):
                                csv_i += 1
                            if df['TransactionID'][csv_i] != trans_id:
                                csv_i = csv_i_old
                                ignored_packets += 1
                                continue

                            attack_type = format_attack(df['Attack'][csv_i])

                    else:
                        attack_type = os.path.splitext(filename)[0]

                    if not render_csv_row(pkt_scapy, macs[i], ether_pkt_sc, ip_pkt_sc, tcp_pkt_sc, modbus_pkt_sc, 1,
                                          attack_type, output_file):
                        ignored_packets += 1
                        f = open("missingThings.txt", "a")
                        f.write("{}:{}\n".format(file, frame_num))
                        f.close()

                    print(
                        'Packets read in current file: {}/1116000 estimated. Files scanned: {}. '
                        'Current file: {}'.format(frame_num, count, file))
                except StopIteration:
                    break

            count += 1
            final_output = ('Progress: {}, {} packets read, {} packets not written to CSV'
                            .format(count, frame_num, ignored_packets))
            outputs.append(final_output)
            print(final_output)

        output_file.close()

    for output in outputs:
        print(output)


if __name__ == '__main__':
    main()
