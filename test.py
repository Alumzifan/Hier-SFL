from scapy.all import *


def test(filepath):
    pcaps = rdpcap(filepath)
    for p in pcaps:
        print(p.show())
        break


filename = r'E:\VPN-PCAPS-01\vpn_aim_chat1a.pcap'
test(filename)
