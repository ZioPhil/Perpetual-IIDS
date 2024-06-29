import os.path
import re
from scapy.utils import RawPcapReader, hexdump
from scapy.layers.l2 import Ether, ARP
from scapy.layers.inet import IP, UDP, TCP, IPOption_Router_Alert
from scapy.layers.inet6 import IPv6, ICMPv6ND_RS, ICMPv6NDOptSrcLLAddr
from scapy.contrib.igmpv3 import IGMPv3, IGMPv3mr, IGMPv3gr
from scapy.layers.dns import DNS, DNSQR
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

fmt_tcp = ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},'
           '{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35}')

fmt_dns = ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},'
           '{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},'
           '{44}')

fmt_arp = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}'

fmt_icmp = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20}'

fmt_igmp = ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},'
            '{23},{24},{25},{26},{27},{28},{29},{30},{31}')

fmt_modbus = ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},'
              '{22},{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},'
              '{42},{43},{44},{45},{46},{47},{48},{49},{50},{51},{52},{53},{54},{55}')

fmt_rmi = ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},'
           '{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38}')


def render_csv_row(pkt_sc, current_mac, f_tcp, f_dns, f_arp, f_icmp, f_igmp, f_modbus, f_rmi, data_type):
    pkt_len = len(pkt_sc)
    ether_pkt_sc = Ether(pkt_sc)
    # print(ether_pkt_sc.show(dump=True))

    timestamp = ether_pkt_sc.time
    src_mac = ether_pkt_sc.src
    dst_mac = ether_pkt_sc.dst

    flow = ""
    if current_mac == src_mac:
        flow = "Sent"
    elif current_mac == dst_mac:
        flow = "Received"

    if ether_pkt_sc.type == 0x800:  # IPv4
        ip_pkt_sc = ether_pkt_sc[IP]
        proto = ip_pkt_sc.proto

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

        if proto == 6:  # TCP
            tcp_pkt_sc = ip_pkt_sc[TCP]

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

            if tcp_pkt_sc.payload:
                if ModbusADUResponse in tcp_pkt_sc or ModbusADURequest in tcp_pkt_sc:
                    if ModbusADUResponse in tcp_pkt_sc:
                        modbus_pkt_sc = tcp_pkt_sc[ModbusADUResponse]
                    else:
                        modbus_pkt_sc = tcp_pkt_sc[ModbusADURequest]

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

                        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac,
                                                dst_mac, src_ip, dst_ip, src_port, dst_port, tcp_chksum, ip_version,
                                                ip_ihl,
                                                ip_tos, ip_len, ip_id, ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq,
                                                tcp_ack, tcp_dataofs, tcp_reserved, tcp_flags, tcp_window, tcp_urgptr,
                                                tcp_option_mss, tcp_option_sackok, tcp_option_timestamp,
                                                tcp_option_timestamp_echoreply, tcp_option_nop_count, tcp_option_wscale,
                                                tcp_option_sack_left, tcp_option_sack_right, modbus_trans_id,
                                                modbus_proto_id, modbus_len, modbus_unit_id, modbus_func_code,
                                                modbus_start_addr, modbus_quantity, "", "", "", "", "", "", "", "", "",
                                                "", "", data_type),
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

                        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac,
                                                dst_mac, src_ip, dst_ip, src_port, dst_port, tcp_chksum, ip_version,
                                                ip_ihl,
                                                ip_tos, ip_len, ip_id, ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq,
                                                tcp_ack, tcp_dataofs, tcp_reserved, tcp_flags, tcp_window, tcp_urgptr,
                                                tcp_option_mss, tcp_option_sackok, tcp_option_timestamp,
                                                tcp_option_timestamp_echoreply, tcp_option_nop_count, tcp_option_wscale,
                                                tcp_option_sack_left, tcp_option_sack_right, modbus_trans_id,
                                                modbus_proto_id, modbus_len, modbus_unit_id, modbus_func_code, "", "",
                                                modbus_output_addr, modbus_output_value, "", "", "", "", "", "", "", "",
                                                "", data_type),
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

                        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac,
                                                dst_mac, src_ip, dst_ip, src_port, dst_port, tcp_chksum, ip_version,
                                                ip_ihl,
                                                ip_tos, ip_len, ip_id, ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq,
                                                tcp_ack, tcp_dataofs, tcp_reserved, tcp_flags, tcp_window, tcp_urgptr,
                                                tcp_option_mss, tcp_option_sackok, tcp_option_timestamp,
                                                tcp_option_timestamp_echoreply, tcp_option_nop_count, tcp_option_wscale,
                                                tcp_option_sack_left, tcp_option_sack_right, modbus_trans_id,
                                                modbus_proto_id, modbus_len, modbus_unit_id, modbus_func_code, "", "",
                                                "", "", modbus_byte_count, modbus_coil_status_0, modbus_coil_status_1,
                                                "", "", "", "", "", "", data_type),
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

                        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac,
                                                dst_mac, src_ip, dst_ip, src_port, dst_port, tcp_chksum, ip_version,
                                                ip_ihl,
                                                ip_tos, ip_len, ip_id, ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq,
                                                tcp_ack, tcp_dataofs, tcp_reserved, tcp_flags, tcp_window, tcp_urgptr,
                                                tcp_option_mss, tcp_option_sackok, tcp_option_timestamp,
                                                tcp_option_timestamp_echoreply, tcp_option_nop_count, tcp_option_wscale,
                                                tcp_option_sack_left, tcp_option_sack_right, modbus_trans_id,
                                                modbus_proto_id, modbus_len, modbus_unit_id, modbus_func_code, "", "",
                                                "", "", modbus_byte_count, "", "", modbus_input_status_0,
                                                modbus_input_status_1, "", "", "", "", data_type),
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

                        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac,
                                                dst_mac, src_ip, dst_ip,
                                                src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len,
                                                ip_id,
                                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                                modbus_unit_id, modbus_func_code, "", "", "", "", modbus_byte_count, "",
                                                "", "", "", modbus_register_val, "", "", "", data_type),
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

                        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac,
                                                dst_mac, src_ip, dst_ip,
                                                src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len,
                                                ip_id,
                                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                                modbus_unit_id, modbus_func_code, "", "", "", "", "", "", "", "",
                                                "", "", modbus_register_addr, modbus_register_value, "",  data_type),
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

                        print(fmt_modbus.format(proto_name, flow, modbus_pdu, modbus_type, timestamp, pkt_len, src_mac,
                                                dst_mac, src_ip, dst_ip,
                                                src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len,
                                                ip_id,
                                                ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                                tcp_reserved, tcp_flags, tcp_window, tcp_urgptr, tcp_option_mss,
                                                tcp_option_sackok, tcp_option_timestamp, tcp_option_timestamp_echoreply,
                                                tcp_option_nop_count, tcp_option_wscale, tcp_option_sack_left,
                                                tcp_option_sack_right, modbus_trans_id, modbus_proto_id, modbus_len,
                                                modbus_unit_id, modbus_func_code, "", "", "", "", "", "", "", "", "",
                                                "", "", "", modbus_except_code, data_type),
                              file=f_modbus)
                        return True

                    else:
                        file = open("missingThings.txt", "a")
                        file.write("Missing MODBUS protocol:\n {}".format(tcp_pkt_sc.show(dump=True)))
                        file.close()
                        return False

                else:
                    rmi_payl = hexdump(bytes(tcp_pkt_sc.payload), dump=True)
                    payl = ""
                    for line in rmi_payl.splitlines():
                        line = re.split(r'\s{2,}', line)[2]
                        payl += line
                    payl = re.sub(r'[.]{2,}', ' ', payl)

                    index = payl.find("1")
                    if index != -1:
                        spl = payl[index:].split(" ")
                        if len(spl) == 12:
                            rmi_type = spl[1]
                            rmi_operation = spl[5]
                            rmi_value = spl[9].split(',')[0]

                            proto_name = 'RMI'
                            print(fmt_rmi.format(proto_name, flow, timestamp, pkt_len, src_mac, dst_mac, src_ip, dst_ip,
                                                 src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len,
                                                 ip_id, ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack,
                                                 tcp_dataofs, tcp_reserved, tcp_flags, tcp_window, tcp_urgptr,
                                                 tcp_option_mss, tcp_option_sackok, tcp_option_timestamp,
                                                 tcp_option_timestamp_echoreply, tcp_option_nop_count,
                                                 tcp_option_wscale, tcp_option_sack_left, tcp_option_sack_right,
                                                 rmi_type, rmi_operation, rmi_value, data_type),
                                  file=f_rmi)
                            return True

                    proto_name = 'TCP'
                    print(fmt_tcp.format(proto_name, flow, timestamp, pkt_len, src_mac, dst_mac, src_ip, dst_ip,
                                         src_port, dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id,
                                         ip_flags, ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs,
                                         tcp_reserved, tcp_flags,
                                         tcp_window, tcp_urgptr, tcp_option_mss, tcp_option_sackok,
                                         tcp_option_timestamp,
                                         tcp_option_timestamp_echoreply, tcp_option_nop_count, tcp_option_wscale,
                                         tcp_option_sack_left, tcp_option_sack_right, data_type),
                          file=f_tcp)
                    return True

            else:
                proto_name = 'TCP'
                print(fmt_tcp.format(proto_name, flow, timestamp, pkt_len, src_mac, dst_mac, src_ip, dst_ip, src_port,
                                     dst_port, tcp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id, ip_flags,
                                     ip_frag, ip_ttl, ip_chksum, tcp_seq, tcp_ack, tcp_dataofs, tcp_reserved, tcp_flags,
                                     tcp_window, tcp_urgptr, tcp_option_mss, tcp_option_sackok, tcp_option_timestamp,
                                     tcp_option_timestamp_echoreply, tcp_option_nop_count, tcp_option_wscale,
                                     tcp_option_sack_left, tcp_option_sack_right, data_type),
                      file=f_tcp)
                return True

        elif proto == 17:  # UDP
            udp_pkt_sc = ip_pkt_sc[UDP]

            src_port = udp_pkt_sc.sport
            dst_port = udp_pkt_sc.dport
            udp_len = udp_pkt_sc.len
            udp_chksum = udp_pkt_sc.chksum

            if udp_pkt_sc[DNS]:
                dns_pkt_sc = udp_pkt_sc[DNS]

                proto_name = 'DNS'
                dns_id = dns_pkt_sc.id
                dns_qr = dns_pkt_sc.qr
                dns_opcode = dns_pkt_sc.opcode
                dns_aa = dns_pkt_sc.aa
                dns_tc = dns_pkt_sc.tc
                dns_rd = dns_pkt_sc.rd
                dns_ra = dns_pkt_sc.ra
                dns_z = dns_pkt_sc.z
                dns_ad = dns_pkt_sc.ad
                dns_cd = dns_pkt_sc.cd
                dns_rcode = dns_pkt_sc.rcode
                dns_qdcount = dns_pkt_sc.qdcount
                dns_ancount = dns_pkt_sc.ancount
                dns_nscount = dns_pkt_sc.nscount
                dns_arcount = dns_pkt_sc.arcount
                dns_an = dns_pkt_sc.an
                dns_ns = dns_pkt_sc.ns
                dns_ar = dns_pkt_sc.ar
                dns_qr_count = 0

                while dns_pkt_sc.payload:
                    dns_pkt_sc = dns_pkt_sc[DNSQR]
                    dns_qr_count += 1

                print(fmt_dns.format(proto_name, flow, timestamp, pkt_len, src_mac, dst_mac, src_ip, dst_ip, src_port,
                                     dst_port, udp_chksum, ip_version, ip_ihl, ip_tos, ip_len, ip_id, ip_flags,
                                     ip_frag, ip_ttl, ip_chksum, "", "", "", "", udp_len, dns_id, dns_qr, dns_opcode,
                                     dns_aa, dns_tc, dns_rd, dns_ra, dns_z, dns_ad, dns_cd, dns_rcode, dns_qdcount,
                                     dns_ancount, dns_nscount, dns_arcount, dns_an, dns_ns, dns_ar, dns_qr_count,
                                     data_type), file=f_dns)
                return True

            else:
                file = open("missingThings.txt", "a")
                file.write("Missing UDP non-DNS packet:\n {}".format(udp_pkt_sc.show(dump=True)))
                file.close()
                return False

        elif proto == 2:  # IGMPv3
            igmp_pkt_sc = ip_pkt_sc[IGMPv3]

            proto_name = 'IGMPv3'

            options = ip_pkt_sc.options[0][IPOption_Router_Alert]
            ip_option_copy_flag = options.copy_flag
            ip_option_optclass = options.optclass
            ip_option_option = options.option
            ip_option_length = options.length
            ip_option_alert = options.alert

            igmp_type = igmp_pkt_sc.type
            igmp_mrcode = igmp_pkt_sc.mrcode
            igmp_chksum = igmp_pkt_sc.chksum

            payload = igmp_pkt_sc[IGMPv3mr]
            igmp_res2 = payload.res2
            igmp_numgrp = payload.numgrp

            records = payload.records[0][IGMPv3gr]
            igmp_rtype = records.rtype
            igmp_auxdlen = records.auxdlen
            igmp_numsrc = records.numsrc
            igmp_maddr = records.maddr

            print(fmt_igmp.format(proto_name, flow, timestamp, pkt_len, src_mac, dst_mac, src_ip, dst_ip, igmp_chksum,
                                  ip_version, ip_ihl, ip_tos, ip_len, ip_id, ip_flags, ip_frag, ip_ttl, ip_chksum,
                                  ip_option_copy_flag, ip_option_optclass, ip_option_option, ip_option_length,
                                  ip_option_alert, igmp_type, igmp_mrcode, igmp_res2, igmp_numgrp, igmp_rtype,
                                  igmp_auxdlen, igmp_numsrc, igmp_maddr, data_type), file=f_igmp)
            return True

        else:
            file = open("missingThings.txt", "a")
            file.write("Missing IP protocol: {}\n".format(proto))
            file.close()
            return False

    elif ether_pkt_sc.type == 0x806:  # ARP
        arp_pkt_sc = ether_pkt_sc[ARP]

        proto_name = 'ARP'
        arp_hwtype = arp_pkt_sc.hwtype
        arp_ptype = arp_pkt_sc.ptype
        arp_hwlen = arp_pkt_sc.hwlen
        arp_plen = arp_pkt_sc.plen
        arp_op = arp_pkt_sc.op
        arp_hwsrc = arp_pkt_sc.hwsrc
        src_ip = arp_pkt_sc.psrc
        arp_hwdst = arp_pkt_sc.hwdst
        dst_ip = arp_pkt_sc.pdst

        print(fmt_arp.format(proto_name, flow, timestamp, pkt_len, src_mac, dst_mac, src_ip, dst_ip, arp_hwtype,
                             arp_ptype, arp_hwlen, arp_plen, arp_op, arp_hwsrc, arp_hwdst, data_type), file=f_arp)
        return True

    elif ether_pkt_sc.type == 0x86dd:  # IPv6
        ipv6_pkt_sc = ether_pkt_sc[IPv6]
        proto = ipv6_pkt_sc.nh

        ip_version = ipv6_pkt_sc.version
        ipv6_tc = ipv6_pkt_sc.tc
        ipv6_fl = ipv6_pkt_sc.fl
        ipv6_plen = ipv6_pkt_sc.plen
        ipv6_hlim = ipv6_pkt_sc.hlim
        src_ip = ipv6_pkt_sc.src
        dst_ip = ipv6_pkt_sc.dst

        if proto == 17:  # UDP
            udp_pkt_sc = ipv6_pkt_sc[UDP]

            src_port = udp_pkt_sc.sport
            dst_port = udp_pkt_sc.dport
            udp_len = udp_pkt_sc.len
            udp_chksum = udp_pkt_sc.chksum

            if udp_pkt_sc[DNS]:
                dns_pkt_sc = udp_pkt_sc[DNS]

                proto_name = 'DNS'
                dns_id = dns_pkt_sc.id
                dns_qr = dns_pkt_sc.qr
                dns_opcode = dns_pkt_sc.opcode
                dns_aa = dns_pkt_sc.aa
                dns_tc = dns_pkt_sc.tc
                dns_rd = dns_pkt_sc.rd
                dns_ra = dns_pkt_sc.ra
                dns_z = dns_pkt_sc.z
                dns_ad = dns_pkt_sc.ad
                dns_cd = dns_pkt_sc.cd
                dns_rcode = dns_pkt_sc.rcode
                dns_qdcount = dns_pkt_sc.qdcount
                dns_ancount = dns_pkt_sc.ancount
                dns_nscount = dns_pkt_sc.nscount
                dns_arcount = dns_pkt_sc.arcount
                dns_an = dns_pkt_sc.an
                dns_ns = dns_pkt_sc.ns
                dns_ar = dns_pkt_sc.ar
                dns_qr_count = 0

                while dns_pkt_sc.payload:
                    dns_pkt_sc = dns_pkt_sc[DNSQR]
                    dns_qr_count += 1 

                print(fmt_dns.format(proto_name, flow, timestamp, pkt_len, src_mac, dst_mac, src_ip, dst_ip, src_port,
                                     dst_port, udp_chksum, ip_version, "", "", "", "", "", "", "", "", ipv6_tc, ipv6_fl,
                                     ipv6_plen, ipv6_hlim, udp_len, dns_id, dns_qr, dns_opcode, dns_aa, dns_tc, dns_rd,
                                     dns_ra, dns_z, dns_ad, dns_cd, dns_rcode, dns_qdcount, dns_ancount, dns_nscount,
                                     dns_arcount, dns_an, dns_ns, dns_ar, dns_qr_count, data_type), file=f_dns)
                return True

            else:
                file = open("missingThings.txt", "a")
                file.write("Missing UDP non-DNS packet:\n {}".format(udp_pkt_sc.show(dump=True)))
                file.close()
                return False

        elif proto == 58:  # ICMPv6
            icmp_pkt_sc = ipv6_pkt_sc[ICMPv6ND_RS]

            proto_name = 'ICMPv6'
            icmp_type = icmp_pkt_sc.type
            icmp_code = icmp_pkt_sc.code
            icmp_chksum = icmp_pkt_sc.cksum
            icmp_res = icmp_pkt_sc.res

            icmp_pkt_sc_opt = icmp_pkt_sc[ICMPv6NDOptSrcLLAddr]
            icmp_opt_type = icmp_pkt_sc_opt.type
            icmp_opt_len = icmp_pkt_sc_opt.len
            icmp_opt_lladr = icmp_pkt_sc_opt.lladdr

            print(fmt_icmp.format(proto_name, flow, timestamp, pkt_len, src_mac, dst_mac, src_ip, dst_ip, icmp_chksum,
                                  ip_version, ipv6_tc, ipv6_fl, ipv6_plen, ipv6_hlim, icmp_type, icmp_code, icmp_res,
                                  icmp_opt_type, icmp_opt_len, icmp_opt_lladr, data_type), file=f_icmp)
            return True

        else:
            file = open("missingThings.txt", "a")
            file.write("Missing IPv6 protocol: {}\n".format(proto))
            file.close()
            return False

    else:
        file = open("missingThings.txt", "a")
        file.write("Missing ETH protocol: {}\n".format(ether_pkt_sc.type))
        file.close()
        return False


def main():
    count = 0
    directories = ["Modbus_Dataset/benign/ied1a",
                   "Modbus_Dataset/benign/ied1b",
                   "Modbus_Dataset/benign/ied4c",
                   "Modbus_Dataset/benign/scada-hmi"]
    output_dirs = ["temp/benignIED1A",
                   "temp/benignIED1B",
                   "temp/benignIED4C",
                   "temp/benignHMI"]
    macs = ["02:42:b9:af:00:04",
            "02:42:b9:af:00:05",
            "02:42:b9:af:00:08",
            "02:42:b9:af:00:03"]
    outputs = []

    for directory in directories:
        i = directories.index(directory)

        files = [f for f in os.listdir(directory)]
        files.sort()
        output_dir = output_dirs[i]

        f_tcp = open("{}/tcp.csv".format(output_dir), "w")
        f_dns = open("{}/dns.csv".format(output_dir), "w")
        f_arp = open("{}/arp.csv".format(output_dir), "w")
        f_icmp = open("{}/icmp.csv".format(output_dir), "w")
        f_igmp = open("{}/igmp.csv".format(output_dir), "w")
        f_modbus = open("{}/modbus.csv".format(output_dir), "w")
        f_rmi = open("{}/rmi.csv".format(output_dir), "w")

        print("proto_name,flow,timestamp,pkt_len,src_mac,dst_mac,src_ip,dst_ip,src_port,dst_port,tcp_chksum,ip_version,"
              "ip_ihl,ip_tos,ip_len,ip_id,ip_flags,ip_frag,ip_ttl,ip_chksum,tcp_seq,tcp_ack,tcp_dataofs,tcp_reserved,"
              "tcp_flags,tcp_window,tcp_urgptr,tcp_option_mss,tcp_option_sackok,tcp_option_timestamp,"
              "tcp_option_timestamp_echoreply,tcp_option_nop_count,tcp_option_wscale,tcp_option_sack_left,"
              "tcp_option_sack_right,data_type", file=f_tcp)

        print("proto_name,flow,timestamp,pkt_len,src_mac,dst_mac,src_ip,dst_ip,src_port,dst_port,udp_chksum,ip_version,"
              "ip_ihl,ip_tos,ip_len,ip_id,ip_flags,ip_frag,ip_ttl,ip_chksum,ipv6_tc,ipv6_fl,ipv6_plen,ipv6_hlim,"
              "udp_len,dns_id,dns_qr,dns_opcode,dns_aa,dns_tc,dns_rd,dns_ra,dns_z,dns_ad,dns_cd,dns_rcode,dns_qdcount,"
              "dns_ancount,dns_nscount,dns_arcount,dns_an,dns_ns,dns_ar,dns_qr_count,data_type", file=f_dns)

        print("proto_name,flow,timestamp,pkt_len,src_mac,dst_mac,src_ip,dst_ip,arp_hwtype,arp_ptype,arp_hwlen,arp_plen,"
              "arp_op,arp_hwsrc,arp_hwdst,data_type", file=f_arp)

        print("proto_name,flow,timestamp,pkt_len,src_mac,dst_mac,src_ip,dst_ip,icmp_chksum,ip_version,ipv6_tc,ipv6_fl,"
              "ipv6_plen,ipv6_hlim,icmp_type,icmp_code,icmp_res,icmp_opt_type,icmp_opt_len,icmp_opt_lladr,data_type",
              file=f_icmp)

        print("proto_name,flow,timestamp,pkt_len,src_mac,dst_mac,src_ip,dst_ip,igmp_chksum,ip_version,ip_ihl,"
              "ip_tos,ip_len,ip_id,ip_flags,ip_frag,ip_ttl,ip_chksum,ip_option_copy_flag,"
              "ip_option_optclass,ip_option_option,ip_option_length,ip_option_alert,igmp_type,igmp_mrcode,igmp_res2,"
              "igmp_numgrp,igmp_rtype,igmp_auxdlen,igmp_numsrc,igmp_maddr,data_type", file=f_igmp)

        print("proto_name,flow,modbus_pdu,modbus_type,timestamp,pkt_len,src_mac,dst_mac,src_ip,dst_ip,src_port,"
              "dst_port,tcp_chksum,ip_version,ip_ihl,"
              "ip_tos,ip_len,ip_id,ip_flags,ip_frag,ip_ttl,ip_chksum,tcp_seq,tcp_ack,tcp_dataofs,tcp_reserved,"
              "tcp_flags,tcp_window,tcp_urgptr,tcp_option_mss,tcp_option_sackok,tcp_option_timestamp,"
              "tcp_option_timestamp_echoreply,tcp_option_nop_count,tcp_option_wscale,tcp_option_sack_left,"
              "tcp_option_sack_right,modbus_trans_id,modbus_proto_id,modbus_len,modbus_unit_id,modbus_func_code,"
              "modbus_start_addr,modbus_quantity,modbus_output_addr,modbus_output_value,modbus_byte_count,"
              "modbus_coil_status_0,modbus_coil_status_1,modbus_input_status_0,modbus_input_status_1,"
              "modbus_register_val,modbus_register_addr,modbus_register_value,modbus_except_code,data_type",
              file=f_modbus)

        print("proto_name,flow,timestamp,pkt_len,src_mac,dst_mac,src_ip,dst_ip,src_port,dst_port,tcp_chksum,ip_version,"
              "ip_ihl,ip_tos,ip_len,ip_id,ip_flags,ip_frag,ip_ttl,ip_chksum,tcp_seq,tcp_ack,tcp_dataofs,tcp_reserved,"
              "tcp_flags,tcp_window,tcp_urgptr,tcp_option_mss,tcp_option_sackok,tcp_option_timestamp,"
              "tcp_option_timestamp_echoreply,tcp_option_nop_count,tcp_option_wscale,tcp_option_sack_left,"
              "tcp_option_sack_right,rmi_type,rmi_operation,rmi_value,data_type", file=f_rmi)

        for file in files:
            file = os.path.join(os.path.abspath(directory), file)

            frame_num = 0
            ignored_packets = 0
            for (pkt_scapy, _) in RawPcapReader(file):
                try:
                    frame_num += 1
                    if not render_csv_row(pkt_scapy, macs[i], f_tcp, f_dns, f_arp, f_icmp, f_igmp, f_modbus, f_rmi, 0):
                        ignored_packets += 1
                        file = open("missingThings.txt", "a")
                        file.write("{}:{}\n".format(file, frame_num))
                        file.close()

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

        f_tcp.close()
        f_dns.close()
        f_arp.close()
        f_icmp.close()
        f_igmp.close()
        f_modbus.close()
        f_rmi.close()

    for output in outputs:
        print(output)


if __name__ == '__main__':
    main()
