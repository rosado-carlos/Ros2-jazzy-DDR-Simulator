import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/carlos/mrad_ws_2609_echo/install/echo_description'
